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


 
# case 2: orion_12CO_mom0.fits
print("PROCESSING: orion_12CO_mom0.fits")
fits_path = '../data/SCIMES/orion_12CO_mom0.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]
    
rms_list = []; inf_list = []; var_list = []; nfl_list = []; nfa_list = []
_rms_list = []; _inf_list = []; _var_list = []; _nfl_list = []; _nfa_list = []
time_list1 = []; time_list2 = []
n_gaussians_list = np.arange(100, 601, 25)
for n_gaussians in n_gaussians_list:
    tic = time.time()
    hdmc = HDMClouds(data, 
                     back_level=1.5, 
                     wcs=wcs, 
                     verbose=False, 
                     n_gaussians=n_gaussians, 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr(max_nfev=10000)
    tac = time.time()
    time_list1.append(hdmc.elapsed_time)
    time_list2.append(tac-tic)
    
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
               "time_gr":time_list1,
               "time_tot":time_list2}

with open('exp-ICE-orionMono.pickle', 'wb') as handle:
    pickle.dump(all_results2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()




