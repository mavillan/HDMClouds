import pickle
import os
import sys
import copy
import numpy as np

# filtering Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

sys.path.append('/user/m/marvill/HDMClouds/lib/')
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


if len(sys.argv)>1:
    out_dir = sys.argv[1]
    if out_dir[-1]!="/": out_dir+="/"
else: out_dir = "./"


# 1) Orion.cont.image.fits
print("PROCESSING: Orion.cont.image.fits")
fits_path = '/user/m/marvill/HDMClouds/data/images/Orion.cont.image.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]

rms_list = []; inf_list = []; var_list = []; nfl_list = []; nfa_list = []
_rms_list = []; _inf_list = []; _var_list = []; _nfl_list = []; _nfa_list = []
time_list = []
alphas = np.arange(0., 5.1, 0.25)
for alpha in alphas:
    hdmc = HDMClouds(data, 
                     back_level=0.089, 
                     wcs=wcs, verbose=False, 
                     n_gaussians=250,
                     alpha=alpha,
                     eps=100., 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr(max_nfev=8000)
    time_list.append(hdmc.elapsed_time)
    
    # computing residuals on evaluation points
    _rmsR, _infR, _varR, _nfa, _nfl = hdmc._get_residual_stats(tuncate=False)
    _rms_list.append(_rmsR)
    _inf_list.append(_infR)
    _var_list.append(_varR)
    _nfa_list.append(_nfa)
    _nfl_list.append(_nfl)
    # computing residual on grid points
    rmsR, infR, varR, nfa, nfl = hdmc.get_residual_stats(truncate=True)
    rms_list.append(rmsR)
    inf_list.append(infR)
    var_list.append(varR)
    nfa_list.append(nfa)
    nfl_list.append(nfl)
    
    del hdmc
all_results = {"rms":rms_list,
               "inf":inf_list,
               "var":var_list,
               "nfa":nfa_list,
               "nfl":nfl_list,
               "_rms":_rms_list,
               "_inf":_inf_list,
               "_var":_var_list,
               "_nfa":_nfa_list,
               "_nfl":_nfl_list,
               "time":time_list}   
with open(out_dir+'exp-alpha-orionKL.pickle', 'wb') as handle:
    pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()



    
# case 2: orion_12CO_mom0.fits
print("PROCESSING: orion_12CO_mom0.fits")
fits_path = '/user/m/marvill/HDMClouds/data/SCIMES/orion_12CO_mom0.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]
    
rms_list = []; inf_list = []; var_list = []; nfl_list = []; nfa_list = []
_rms_list = []; _inf_list = []; _var_list = []; _nfl_list = []; _nfa_list = []
time_list = []
alphas = np.arange(0., 5.1, 0.25)
for alpha in alphas:
    hdmc = HDMClouds(data, 
                     back_level=1.5, 
                     wcs=wcs, 
                     verbose=False, 
                     n_gaussians=400,
                     alpha=alpha,
                     eps=100., 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr(max_nfev=10000)
    time_list.append(hdmc.elapsed_time)
    
    # computing residuals on evaluation points
    _rmsR, _infR, _varR, _nfa, _nfl = hdmc._get_residual_stats(truncate=False)
    _rms_list.append(_rmsR)
    _inf_list.append(_infR)
    _var_list.append(_varR)
    _nfa_list.append(_nfa)
    _nfl_list.append(_nfl)
    # computing residual on grid points
    rmsR, infR, varR, nfa, nfl = hdmc.get_residual_stats(truncate=True)
    rms_list.append(rmsR)
    inf_list.append(infR)
    var_list.append(varR)
    nfa_list.append(nfa)
    nfl_list.append(nfl)
    
    del hdmc
all_results2 = {"rms":rms_list,
               "inf":inf_list,
               "var":var_list,
               "nfa":nfa_list,
               "nfl":nfl_list,
               "_rms":_rms_list,
               "_inf":_inf_list,
               "_var":_var_list,
               "_nfa":_nfa_list,
               "_nfl":_nfl_list,
               "time":time_list}

with open(out_dir+'exp-alpha-orionMono.pickle', 'wb') as handle:
    pickle.dump(all_results2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    


# case 3: Orion.methanol.cbc.contsub.image.fits
print("PROCESSING: Orion.methanol.cbc.contsub.image.fits")
fits_path = '/user/m/marvill/HDMClouds/data/cubes/Orion.methanol.cbc.contsub.image.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]
spec = loaded_fits["spec"]
base_level = estimate_rms(data)

rms_list = []; inf_list = []; var_list = []; nfl_list = []; nfa_list = []
_rms_list = []; _inf_list = []; _var_list = []; _nfl_list = []; _nfa_list = []
time_list = []
alphas = np.arange(0., 5.1, 0.25)
for alpha in alphas:
    hdmc = HDMClouds(data, 
                     back_level=base_level, 
                     freq=spec, 
                     wcs=wcs, 
                     verbose=False, 
                     n_gaussians=800,
                     alpha=alpha,
                     eps=100., 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr(max_nfev=50000)
    time_list.append(hdmc.elapsed_time)
    
    # computing residuals on evaluation points
    _rmsR, _infR, _varR, _nfa, _nfl = hdmc.get_residual_stats(truncate=False)
    _rms_list.append(_rmsR)
    _inf_list.append(_infR)
    _var_list.append(_varR)
    _nfa_list.append(_nfa)
    _nfl_list.append(_nfl)
    # computing residual on grid points
    rmsR, infR, varR, nfa, nfl = hdmc.get_residual_stats(truncate=True)
    rms_list.append(rmsR)
    inf_list.append(infR)
    var_list.append(varR)
    nfa_list.append(nfa)
    nfl_list.append(nfl)

    del hdmc
all_results3 = {"rms":rms_list,
               "inf":inf_list,
               "var":var_list,
               "nfa":nfa_list,
               "nfl":nfl_list,
               "_rms":_rms_list,
               "_inf":_inf_list,
               "_var":_var_list,
               "_nfa":_nfa_list,
               "_nfl":_nfl_list,
               "time":time_list}

with open(out_dir+'exp-alpha-orionKLCube.pickle', 'wb') as handle:
    pickle.dump(all_results3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
