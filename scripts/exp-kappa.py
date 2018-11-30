import pickle
import os
import sys
import copy
import numpy as np

# filtering Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

sys.path.append('../lib/')
from hdmclouds_v1 import *
from utils import *
from graph import *
from preprocessing import *
from gmr import *

from ipywidgets import interact, interactive, fixed, FloatSlider, IntSlider
from IPython.display import display

import matplotlib
import matplotlib.pyplot as plt; plt.show()
matplotlib.rcParams.update({'font.size': 15})


# 1) Orion.cont.image.fits
print("PROCESSING: Orion.cont.image.fits")
fits_path = '../data/images/Orion.cont.image.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]

all_results = dict()
kappa_values = np.arange(0.1, 5.1, 0.1)
for n_gaussians in range(200, 401, 50):
    res_rms_list = []; res_inf_list = []; res_var_list = []; res_nfa_list = []; res_fl_list = []
    _res_rms_list = []; _res_inf_list = []; _res_var_list = []; _res_nfa_list = []; _res_fl_list = []
    time_list = []
    for kappa in kappa_values:
        hdmc = HDMClouds(data, 
                         back_level=0.089, 
                         wcs=wcs, 
                         verbose=False, 
                         n_gaussians=n_gaussians, 
                         eps=100., 
                         kappa=kappa, 
                         gmr_neighbors=64)
        hdmc.build_gmr(max_nfev=5000)
        # computing residual stats
        _res_rms,_res_inf,_res_var,_res_nfa,_res_fl = hdmc._get_residual_stats()
        res_rms,res_inf,res_var,res_nfa,res_fl = hdmc.get_residual_stats()
        # residuals computed in evaluation points
        _res_rms_list.append(_res_rms)
        _res_inf_list.append(_res_inf)
        _res_var_list.append(_res_var)
        _res_nfa_list.append(_res_nfa)
        _res_fl_list.append(_res_fl)
        # residuals computed in grid points
        res_rms_list.append(res_rms)
        res_inf_list.append(res_inf)
        res_var_list.append(res_var)
        res_nfa_list.append(res_nfa)
        res_fl_list.append(res_fl)
        # elapsed time
        time_list.append(hdmc.elapsed_time)
        del hdmc
    all_results[str(n_gaussians)+"_gs"] = [(res_rms_list, res_inf_list, res_inf_list, res_nfa_list, res_fl_list, time_list),
                                           (_res_rms_list, _res_inf_list, _res_inf_list, _res_nfa_list, _res_fl_list, time_list)]

with open('exp-kappa-orionKL.pickle', 'wb') as handle:
    pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()



# case 2: orion_12CO_mom0.fits
print("PROCESSING: orion_12CO_mom0.fits")
fits_path = '../data/SCIMES/orion_12CO_mom0.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]

all_results2 = dict()
kappa_values = np.arange(0.1, 5.1, 0.1)
for n_gaussians in range(200, 601, 50):
    res_rms_list = []; res_inf_list = []; res_var_list = []; res_nfa_list = []; res_fl_list = []
    _res_rms_list = []; _res_inf_list = []; _res_var_list = []; _res_nfa_list = []; _res_fl_list = []
    time_list = []
    for kappa in kappa_values:
        hdmc = HDMClouds(data, 
                         back_level=1.5, 
                         wcs=wcs, 
                         verbose=False, 
                         n_gaussians=n_gaussians, 
                         eps=100., 
                         kappa=kappa, 
                         gmr_neighbors=64)
        hdmc.build_gmr(max_nfev=6000)
        # computing residual stats
        _res_rms,_res_inf,_res_var,_res_nfa,_res_fl = hdmc._get_residual_stats()
        res_rms,res_inf,res_var,res_nfa,res_fl = hdmc.get_residual_stats()
        # residuals computed in evaluation points
        _res_rms_list.append(_res_rms)
        _res_inf_list.append(_res_inf)
        _res_var_list.append(_res_var)
        _res_nfa_list.append(_res_nfa)
        _res_fl_list.append(_res_fl)
        # residuals computed in grid points
        res_rms_list.append(res_rms)
        res_inf_list.append(res_inf)
        res_var_list.append(res_var)
        res_nfa_list.append(res_nfa)
        res_fl_list.append(res_fl)
        # elapsed time
        time_list.append(hdmc.elapsed_time)
        del hdmc
    all_results2[str(n_gaussians)+"_gs"] = [(res_rms_list, res_inf_list, res_inf_list, res_nfa_list, res_fl_list, time_list),
                                           (_res_rms_list, _res_inf_list, _res_inf_list, _res_nfa_list, _res_fl_list, time_list)]

with open('exp-kappa-orionMono.pickle', 'wb') as handle:
    pickle.dump(all_results2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()



# case 3: Orion.methanol.cbc.contsub.image.fits
print("PROCESSING: Orion.methanol.cbc.contsub.image.fits")
fits_path = '../data/cubes/Orion.methanol.cbc.contsub.image.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]
spec = loaded_fits["spec"]
base_level = estimate_rms(data)

all_results3 = dict()
kappa_values = np.arange(0.5, 5.1, 0.5)
for n_gaussians in range(500, 1001, 50):
    res_rms_list = []; res_inf_list = []; res_var_list = []; res_nfa_list = []; res_nfl_list = []
    _res_rms_list = []; _res_inf_list = []; _res_var_list = []; _res_nfa_list = []; _res_nfl_list = []
    time_list = []
    for kappa in kappa_values:
        hdmc = HDMClouds(data, 
                         back_level=base_level, 
                         wcs=wcs, 
                         verbose=False, 
                         n_gaussians=n_gaussians, 
                         eps=100., 
                         kappa=kappa, 
                         gmr_neighbors=64)
        hdmc.build_gmr(max_nfev=20000)
        # computing residual stats
        _res_rms,_res_inf,_res_var,_res_nfa,_res_nfl = hdmc._get_residual_stats()
        res_rms,res_inf,res_var,res_nfa,res_nfl = hdmc.get_residual_stats()
        # residuals computed in evaluation points
        _res_rms_list.append(_res_rms)
        _res_inf_list.append(_res_inf)
        _res_var_list.append(_res_var)
        _res_nfa_list.append(_res_nfa)
        _res_nfl_list.append(_res_nfl)
        # residuals computed in grid points
        res_rms_list.append(res_rms)
        res_inf_list.append(res_inf)
        res_var_list.append(res_var)
        res_nfa_list.append(res_nfa)
        res_nfl_list.append(res_nfl)
        # elapsed time
        time_list.append(hdmc.elapsed_time)
        del hdmc
    all_results3[str(n_gaussians)+"_gs"] = [(res_rms_list, res_inf_list, res_inf_list, res_nfa_list, res_nfl_list, time_list),
                                           (_res_rms_list, _res_inf_list, _res_inf_list, _res_nfa_list, _res_nfl_list, time_list)]

with open('exp-kappa-orionKLCube.pickle', 'wb') as handle:
    pickle.dump(all_results3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()






