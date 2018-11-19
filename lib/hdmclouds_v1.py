import time
import string
import copy
import scipy
import numba
import scipy as sp
import numpy as np
import pandas as pd
from math import sqrt, exp
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

import graph as gp
from utils import *
from points_generation import *
from preprocessing import *
from gmr import *
from fgm_eval import *

from IPython.display import display




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
    def __init__(self, data, freq=None, n_gaussians=250, alpha=0., lamb=1., 
                 back_level=None, minsig=None, maxsig=None, kappa=5., eps=None,
                 bound_spacing=None, gmr_neighbors=None, verbose=False, wcs=None):

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
        self.n_gaussians = n_gaussians
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
        compression = n_gaussians/float(npix)
        print("[INFO] Number of pixels with significant emission: {0}".format(npix))
        print("[INFO] Level of compression: {0}%".format(compression*100))
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
        if self.ndim==2:
            if bound_spacing is None:
                bound_spacing = 2.5*pix_lenght
            bound_points = boundary_points_generation(data, mask, bound_spacing)
            xb = bound_points[:,0]; yb = bound_points[:,1]
        if self.ndim==3:
            xb = None; yb = None; zb = None
            bound_points = None

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
        #if self.ndim==2: pix_lenght_ = sum([1./data.shape[0], 1./data.shape[1]])/2.
        #if self.ndim==3: pix_lenght_ = sum([1./data.shape[0], 1./data.shape[1], 1./data.shape[2]])/3.
        sig_init = (pix_lenght/(2.*k))*np.ones(w_init.shape[0])

        # DBSAN to find isolated structure
        if eps is None:
            if self.ndim==2: eps = np.sqrt(2)*pix_lenght
            if self.ndim==3: eps = np.sqrt(3)*pix_lenght
        db = DBSCAN(eps=eps, min_samples=4, n_jobs=-1)
        db.fit(mu_init)
        db_labels = db.labels_ # predicted labels
        print("[INFO] Number of ICEs: {0}".format(db_labels.max()+1))

        # we train a KNN classifier with all the assigned points
        assigned = mu_init[db_labels!=-1]
        assigned_labels = db_labels[db_labels!=-1]
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(assigned, assigned_labels)

        unassigned = mu_init[db_labels==-1]
        if len(unassigned)>0:
            print("[INFO] There are unassigned pixels: Automatically assigning to closer ICE\n")
            unassigned_labels = clf.predict(unassigned)
            db_labels[db_labels==-1] = unassigned_labels

        # splitting boundary points by ICE
        if bound_points is not None:
            bp_labels = clf.predict(bound_points)
        else: bp_labels = None

        # instantiating HDICE for each Isolated Cloud Entity
        hdice_list = list()
        hdice_dict = dict()
        num_ice = db_labels.max()+1

        alph = string.ascii_uppercase
        if num_ice<=26:    hdice_keys = [alph[i%26] for i in range(num_ice)]
        elif num_ice<=676: hdice_keys = [alph[i//26]+alph[i%26] for i in range(num_ice)]
        else:              hdice_keys=  [alph[i//(26*26)]+alph[(i//26)%26]+alph[i%26] for i in range(num_ice)]
        
        for i in range(0, db_labels.max()+1):
            _mask = db_labels==i
            if bp_labels is not None: _bmask = bp_labels==i
            print("Isolated Cloud Entity {0}: {1} pixels of significant emission.".format(hdice_keys[i], np.sum(_mask)))

            if self.ndim==2:
                hdice = HDICE(w_init[_mask], mu_init[_mask], sig_init[_mask], 
                              back_level, alpha, lamb, compression, kappa=kappa,
                              gmr_neighbors=gmr_neighbors,
                              xgrid_global=xgrid, ygrid_global=ygrid, 
                              bound_points=bound_points[_bmask])
            if self.ndim==3:
                hdice = HDICE(w_init[_mask], mu_init[_mask], sig_init[_mask], 
                              back_level, alpha, lamb, compression, kappa=kappa,
                              gmr_neighbors=gmr_neighbors,
                              xgrid_global=xgrid, ygrid_global=ygrid, zgrid_global=zgrid, 
                              bound_points=None)

            hdice.set_f0(dfunc(hdice.eval_points))
            if hdice.bound_points is not None: hdice.set_fb(dfunc(hdice.bound_points))
            hdice.set_fgrid(dfunc(hdice.grid_points))
            hdice_list.append(hdice)
            hdice_dict[hdice_keys[i]] = hdice

        # visualizing the results of ICE
        if verbose:
            if self.ndim==2: 
                gp.points_clusters(data, mu_init, db_labels, hdice_keys, 
                                   wcs=wcs, title="Isolated Cloud Entities")
            if self.ndim==3:
                pass

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
                if bound_points is not None:
                    gp.points_plot(data, points=bound_points, color="green", wcs=wcs, title="Boundary Points")
            if self.ndim==3:
                pass

        ########################################
        # HDMClouds internals
        ########################################
        # list and dict with HDICE objetcs for each ICE
        self.hdice_list = hdice_list
        self.hdice_dict = hdice_dict
        self.hdice_keys = hdice_keys
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

    def coord2world(self, xcoord, ycoord, zcoord=None):
        # we first transform from our internal coordinate
        # to a pixel position
        px = (xcoord-self.pix_lenght/2.)/self.pix_lenght
        py = (ycoord-self.pix_lenght/2.)/self.pix_lenght
        if self.ndim==3: pz = (zcoord-self.pix_lenght/2.)/self.pix_lenght
            
        if self.ndim==2: 
            positions = np.vstack([px,py]).T
            converted = self.wcs.wcs_pix2world(positions,0)
            angx = converted[:,0]
            angy = converted[:,1]
        if self.ndim==3: 
            positions = np.vstack([px,py,pz]).T
            converted = self.wcs.wcs_pix2world(positions,0)
            angx = converted[:,0]
            angy = converted[:,1]
            freq = converted[:,2]
            # from Hz to Ghz 
            freq = ["{:.5f} GHz".format(f/10**9) for f in freq] 
        
        # convinient transformations
        angx = Angle(angx, units.degree)
        angy = Angle(angy, units.degree)
        angx = angx.to_string(unit=units.degree, sep=('deg', 'm', 's'))
        angy = angy.to_string(unit=units.degree, sep=('deg', 'm', 's'))
        
        if self.ndim==2: return angx,angy
        if self.ndim==3: return angx,angy,freq

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

    def get_residual_stats(self, verbose=True):
        u = self.get_approximation()
        if self.mask is not None:
            residual = np.zeros(self.shape)
            residual[self.mask] = self.data[self.mask]-u[self.mask]
        else:
            residual = self.data-u

        # visualization original data, u and residual
        if verbose and self.ndim==2:
            gp.solution_plot(self.data, u, residual)
            gp.residual_histogram(residual[self.mask])

        if verbose and self.ndim==3:
            print("-"*110)
            print("ORIGINAL DATA")
            gp.cube_plot(self.data, wcs=self.wcs, freq=self.freq)
            print("-"*110)
            print("GAUSSIAN MIXTURE")
            gp.cube_plot(u, wcs=self.wcs, freq=self.freq)
            print("-"*110)
            print("RESIDUAL")
            gp.cube_plot(residual, wcs=self.wcs, cmap=plt.cm.RdBu_r, freq=self.freq)
            print("-"*110)

        # computing residual stats
        total_flux = np.sum(self.data[self.mask])
        flux_mask = residual<0.
        flux_addition = -1. * np.sum(residual[flux_mask])
        flux_lost = np.sum(residual[~flux_mask])

        # output residuals
        out = (estimate_rms(residual), np.max(np.abs(residual)), estimate_variance(residual), \
               flux_addition/total_flux, flux_lost/total_flux)
        
        # printing output
        if verbose:
            print("RESIDUAL STATS")
            print("RMS of residual: {0}".format(out[0]))
            print("Inf norm of residual: {0}".format(out[1]))
            print("Variance of residual: {0}".format(out[2]))
            print("Normalized flux addition: {0}".format(out[3]))
            print("Normalized flux lost: {0}".format(out[4]))
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
            print("Building GM for Isolated Cloud Entity {0}".format(self.hdice_keys[i]))
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

    def build_hierarchical_tree(self, htree_algo="KL"):
        for i,hdice in enumerate(self.hdice_list):
            print("Building the hierarchical tree for Isolated Cloud Entity {0}".format(self.hdice_keys[i]))
            hdice.build_htree(htree_algo=htree_algo)
            print("DONE\n")

        # global list of splittable and joinable cloud entities
        splittable = list()
        joinable = list()
        for ice_key in self.hdice_dict.keys():
            splittable.append(ice_key+"-0")

        # actual values
        self.splittable = sorted(splittable)
        self.joinable = sorted(joinable)

        # original values are stored for the reset
        self.splittable_reset = sorted(splittable)
        self.joinable_reset = sorted(joinable)

    def split_ce(self, CEid):
        """
        CEid (string) - Identifier of the cloud entity to be splitted: "A-0"
        """
        ice_key,idx = CEid.split("-")
        if ice_key in self.hdice_dict:
            hdice = self.hdice_dict[ice_key]
        else: 
            print("Invalid CEid")
            return None

        if int(idx) in hdice.decomp_dict:
            idx1,idx2 = hdice.decomp_dict[int(idx)]
        else:
            print("Invalid CEid")
            return None

        # updating splittable cloud entities
        self.splittable.remove(CEid)
        self.splittable.append(ice_key+"-"+str(idx1))
        self.splittable.append(ice_key+"-"+str(idx2))
        self.splittable.sort()

        # updating joinable cloud entities
        _joinable = copy.copy(self.joinable)
        for i,tup in enumerate(self.joinable):
            if CEid in tup: del _joinable[i]
            # podria ir un break?
        _joinable.append( (ice_key+"-"+str(idx1),ice_key+"-"+str(idx2)) )
        self.joinable = sorted(_joinable)

    def join_ce(self, CEid_tuple):
        CEid1,CEid2 = CEid_tuple
        ice_key,idx1 = CEid1.split("-")
        _,idx2 = CEid2.split("-")

        if ice_key in self.hdice_dict:
            hdice = self.hdice_dict[ice_key]
        else:
            print("Invalid CEid_tuple")
            return None 

        if (int(idx1),int(idx2)) in hdice.join_dict:
            idx = hdice.join_dict[(int(idx1),int(idx2))]
        elif (int(idx2),int(idx1)) in hdice.join_dict:
            idx = hdice.join_dict[(int(idx2),int(idx1))]
        else:
            print("Invalid CEid_tuple")
            return None


        # updating splittable cloud entities
        self.splittable.remove(CEid1)
        self.splittable.remove(CEid2)
        self.splittable.append(ice_key+"-"+str(idx))
        self.splittable.sort()

        # updating joinable cloud entities
        self.joinable.remove(CEid_tuple)
        for tup in hdice.join_dict:
            if idx in tup:
                idx1,idx2 = tup
                self.joinable.append( (ice_key+"-"+str(idx1),ice_key+"-"+str(idx2)) )
                break
        self.joinable.sort()

    def reset_hierarchical_tree(self):
        self.splittable = self.splittable_reset
        self.joinable = self.joinable_reset

    def compute_stats(self):
        stats = dict()

        for CEid in self.splittable:
            ice_key,idx = CEid.split("-")
            hdice = self.hdice_dict[ice_key]

            indexes = hdice.entity_dict[int(idx)]
            params = hdice.get_params_filtered(indexes)
            if self.ndim==2: xc,yc,w,sig = params
            if self.ndim==3: xc,yc,zc,w,sig = params

            xe,ye = hdice.xe,hdice.ye
            if self.ndim==3: ze = hdice.ze
            nn_ind1 = hdice.nn_ind1
            nn_ind1_aux = hdice.nn_ind1_aux

            if self.ndim==2: u = gm_eval2d_2(w, sig, xc, yc, xe, ye, nn_ind1, nn_ind1_aux)
            if self.ndim==3: u = gm_eval3d_2(w, sig, xc, yc, zc, xe, ye, ze, nn_ind1, nn_ind1_aux)

            # we first compute the centroid
            x_centroid = np.sum(u*xe)/np.sum(u)
            y_centroid = np.sum(u*ye)/np.sum(u)
            if self.ndim==3: z_centroid = np.sum(u*ze)/np.sum(u)

            if self.ndim==2: 
                out = self.coord2world(y_centroid,x_centroid)
                ra_centroid,dec_centroid = out
            if self.ndim==3: 
                out = self.coord2world(y_centroid,x_centroid,z_centroid)
                ra_centroid,dec_centroid,f_centroid = out

            # we apply to u the inverse mapping
            total_flux = np.sum((self.vmax-self.vmin)*u + self.vmin)

            if self.ndim==2: stats[CEid] = (total_flux,
                                            ra_centroid[0], 
                                            dec_centroid[0])
            if self.ndim==3: stats[CEid] = (total_flux, 
                                            ra_centroid[0], 
                                            dec_centroid[0], 
                                            f_centroid[0])

        if self.ndim==2:
            stats = pd.DataFrame.from_dict(
                    stats,
                    orient="index",
                    columns=["Flux [Jy/Beam]",
                             "Right Ascension",
                             "Declination"])
        if self.ndim==3:
            stats = pd.DataFrame.from_dict(
                    stats,
                    orient="index",
                    columns=["Flux [Jy/Beam]",
                             "Right Ascension",
                             "Declination",
                             "Frequency"])
        return stats

    
    def visualize(self):
        def handler(hdmc, split="", join1="", join2="", reset=False, show_stats=False):
            if reset:
                hdmc.reset_hierarchical_tree()
            elif len(split)!=0:
                if split in hdmc.splittable:
                    hdmc.split_ce(split)
            elif len(join1)!=0 and len(join2)!=0:
                if (join1,join2) in hdmc.joinable:
                    hdmc.join_ce((join1,join2))
                elif (join2,join1) in hdmc.joinable:
                    hdmc.join_ce((join2,join1))
            if hdmc.ndim==2: gp.ce_plot(hdmc, wcs=hdmc.wcs)
            if hdmc.ndim==3: gp.ce_plot_3d(hdmc, wcs=hdmc.wcs)

            if show_stats:
                stats = self.compute_stats()
                display(stats)
            return None

        interact(handler, hdmc=fixed(self), split="", join1="", join2="", reset=False, show_stats=False);



class HDICE():
    """
    Hierarchical Decomposition of Independent/Isolated Cloud Entities
    """
    def __init__(self, w_init, mu_init, sig_init, back_level, alpha, lamb, compression, 
                 minsig=None, maxsig=None, kappa=5., gmr_neighbors=None, verbose=False, 
                 xgrid_global=None, ygrid_global=None, zgrid_global=None, bound_points=None, 
                 min_num_gaussians=5):

        self.ndim = mu_init.shape[1]
        # Max intensity in the CE
        data_max = w_init.max()

        # local grid points
        xgrid = mu_init[:,0]
        ygrid = mu_init[:,1]
        if self.ndim==3: zgrid = mu_init[:,2]
        grid_points = mu_init

        # global grid points
        if self.ndim==2: grid_points_global = np.vstack([xgrid_global,ygrid_global]).T
        if self.ndim==3: grid_points_global = np.vstack([xgrid_global,ygrid_global,zgrid_global]).T

        # boundary points
        if bound_points is not None:
            xb = bound_points[:,0]
            yb = bound_points[:,1]
            if self.ndim==3: zb = bound_points[:,2]
        else:
            xb = None; yb = None; zb = None

        # target number of gaussians
        n_gaussians = max(int(len(w_init)*compression), min_num_gaussians)

        w_red, mu_red, cov_red = reduce_mixture(w_init, mu_init, sig_init, 2*n_gaussians, 
                                                n_neighbors=gmr_neighbors, verbose=False)
        xe = mu_red[:,0]
        ye = mu_red[:,1]
        if self.ndim==3: ze = mu_red[:,2]
        eval_points = mu_red

        w, mu, cov = reduce_mixture(w_red, mu_red, cov_red, n_gaussians, 
                                    n_neighbors=gmr_neighbors, verbose=False)
        xc = mu[:,0]
        yc = mu[:,1]
        if self.ndim==3: zc = mu[:,2]
        center_points = mu

        # truncation of the covariance matrices
        sig = np.asarray( [np.mean((np.linalg.eig(_cov)[0]))**(1./2) for _cov in cov] )

        minsig = np.min(np.abs(sig))
        maxsig = 3*np.max(np.abs(sig))
        epsilon = 1e-6 # little shift to avoid NaNs in inv_sig_mapping

        #######################################
        # Computing neighborhoods
        #######################################

        nn_indexes,nn_indexes_aux = compute_neighbors(center_points, eval_points, kappa*maxsig)
        self.nn_ind1 = nn_indexes
        self.nn_ind1_aux = nn_indexes_aux

        if bound_points is not None:
            nn_indexes,nn_indexes_aux = compute_neighbors(center_points, bound_points, kappa*maxsig)
            self.nn_ind2 = nn_indexes
            self.nn_ind2_aux = nn_indexes_aux
        else:
            self.nn_ind2 = None
            self.nn_ind2_aux = None

        nn_indexes,nn_indexes_aux = compute_neighbors(center_points, grid_points, kappa*maxsig)
        self.nn_ind3 = nn_indexes
        self.nn_ind3_aux = nn_indexes_aux

        nn_indexes,nn_indexes_aux = compute_neighbors(center_points, grid_points_global, kappa*maxsig)
        self.nn_ind4 = nn_indexes
        self.nn_ind4_aux = nn_indexes_aux

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
        self.fb = None
        self.fgrid = None

        # center points
        self.xc = xc; self.yc = yc 
        if self.ndim==3: self.zc = zc
        self.center_points = center_points

        # boundary points
        self.xb = xb; self.yb = yb
        if self.ndim==3: self.zb = zb
        self.bound_points = bound_points

        # evaluation points
        self.xe = xe; self.ye = ye
        if self.ndim==3: self.ze = ze
        self.eval_points = eval_points

        # local grid points
        self.xgrid = xgrid; self.ygrid = ygrid
        if self.ndim==3: self.zgrid = zgrid
        self.grid_points = grid_points

        # global grid points
        self.xgrid_global = xgrid_global
        self.ygrid_global = ygrid_global
        if self.ndim==3: self.zgrid_global = zgrid_global
        self.grid_points_global = grid_points_global

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
        self.alpha = alpha
        self.lamb = lamb
        self.back_level = back_level
        self.scipy_sol = None
        self.scipy_tol = None
        self.elapsed_time = None
        self.residual_stats = None

        # hierarchical tree data structures
        self.decomp_dict = None
        self.join_dict = None
        self.entity_dict = None

    def set_f0(self, f0):
        self.f0 = f0

    def set_fb(self, fb):
        self.fb = fb

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

    def get_params_filtered(self, indexes):
        """
        Some explanation
        """
        w,sig = self.get_params_mapped()
        xc = self.xc; yc = self.yc
        if self.ndim==3: zc = self.zc
        N = len(w)
        ret_xc = np.zeros(N)
        ret_yc = np.zeros(N)
        if self.ndim==3: ret_zc = np.zeros(N)
        ret_w = np.zeros(N)
        ret_sig = np.zeros(N)

        ret_xc[indexes] = xc[indexes]
        ret_yc[indexes] = yc[indexes]
        if self.ndim==3: ret_zc[indexes] = zc[indexes]
        ret_w[indexes] = w[indexes]
        ret_sig[indexes] = sig[indexes]
        
        if self.ndim==2: return ret_xc,ret_yc,ret_w,ret_sig
        if self.ndim==3: return ret_xc,ret_yc,ret_zc,ret_w,ret_sig

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

    def get_approximation(self, params=None):
        if params is None:
            w,sig = self.get_params_mapped()
            xc = self.xc; yc = self.yc
            if self.ndim==3: zc = self.zc
        else:
            if self.ndim==2: xc,yc,w,sig = params
            if self.ndim==3: xc,yc,zc,w,sig = params

        if self.ndim==2:
            u = gm_eval2d_2(w, sig, xc, yc, self.xgrid, self.ygrid, self.nn_ind3, self.nn_ind3_aux)
        if self.ndim==3:
            u = gm_eval3d_2(w, sig, xc, yc, zc, self.xgrid, self.ygrid, self.zgrid, self.nn_ind3, self.nn_ind3_aux)
        return u

    def get_approximation_global(self, params=None):
        if params is None:
            w,sig = self.get_params_mapped()
            xc = self.xc; yc = self.yc
            if self.ndim==3: zc = self.zc
        else:
            if self.ndim==2: xc,yc,w,sig = params
            if self.ndim==3: xc,yc,zc,w,sig = params

        if self.ndim==2:
            u = gm_eval2d_2(w, sig, xc, yc, self.xgrid_global, self.ygrid_global, self.nn_ind4, self.nn_ind4_aux)
        if self.ndim==3:
            u = gm_eval3d_2(w, sig, xc, yc, zc, self.xgrid_global, self.ygrid_global, self.zgrid_global, self.nn_ind4, self.nn_ind4_aux)
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

        # evaluation of the EL equation (2*(u-f0) + alpha*d1psi1(u-f0))
        alpha = self.alpha; lamb = self.lamb
        el = u-self.f0
        if alpha>0.: flux_term = alpha*d1psi1(el, lamb)
        el *= 2
        if alpha>0.: el += flux_term

        # evaluating at the boundary, if not None
        if self.bound_points is not None:
            fb = self.fb
            if self.ndim==2:
                u_boundary = gm_eval2d_2(w, sig, self.xc, self.yc, self.xb, self.yb, self.nn_ind2, self.nn_ind2_aux)
            if self.ndim==3:
                u_boundary = gm_eval3d_2(w, sig, self.xc, self.yc, self.zc, self.xb, self.yb, self.zb, self.nn_ind2, self.nn_ind2_aux)
            return np.concatenate([el,u_boundary-fb])
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

    def build_htree(self, htree_algo="KL", n_neighbors=None):
        w,sig = self.get_params_mapped()
        if self.ndim==2: mu = np.vstack([self.xc,self.yc]).T
        if self.ndim==3: mu = np.vstack([self.xc,self.yc,self.zc]).T
        if n_neighbors is None: n_neighbors = self.gmr_neighbors
        
        if htree_algo=="KL":
            htree = agglomerate_kl(w, mu, sig, n_neighbors=n_neighbors, verbose=False)
        elif htree_algo=="ISD":
            htree = agglomerate_isd(w, mu, sig, verbose=False)
        decomp_dict,join_dict,entity_dict = htree

        # dictionary to properly map the identifiers of the CE
        # in the decomp_dict, join_dict and entity_dict
        mp = dict()
        for i,key in enumerate(sorted(entity_dict.keys(), reverse=True)): mp[key] = i

        _decomp_dict = dict()
        _join_dict = dict()
        _entity_dict = dict()

        for key,value in decomp_dict.items():
            id1,id2 = value
            _decomp_dict[mp[key]] = (mp[id1],mp[id2])

        for key,value in join_dict.items():
            id1,id2 = key
            _join_dict[(mp[id1],mp[id2])] = mp[value]

        for key,value in entity_dict.items():
            _entity_dict[mp[key]] = value

        self.decomp_dict = _decomp_dict
        self.join_dict = _join_dict
        self.entity_dict = _entity_dict

        return (_decomp_dict,_join_dict,_entity_dict)


