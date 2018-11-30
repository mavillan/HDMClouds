import pickless
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

rms_list = []
inf_list = []
var_list = []
nfl_list = []
nfa_list = []
time_list = []
n_gaussians_list = np.arange(50, 401, 25)
for n_gaussians in n_gaussians_list:
    hdmc = HDMClouds(data, 
                     back_level=0.089, 
                     wcs=wcs, verbose=False, 
                     n_gaussians=n_gaussians, 
                     eps=100., 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr()
    rmsR, infR, varR, nfa, nfl = hdmc.get_residual_stats(verbose=False)
    rms_list.append(rmsR)
    inf_list.append(infR)
    var_list.append(varR)
    nfa_list.append(nfa)
    nfl_list.append(nfl)
    time_list.append(hdmc.elapsed_time)
    del hdmc
all_results = {"rms":rms_list,
               "inf":inf_list,
               "var":var_list,
               "nfa":nfa_list,
               "nfl":nfl_list,
               "time":time_list}   
with open('exp-n_gaussians-orionKL.pickle', 'wb') as handle:
    pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()



    
# case 2: orion_12CO_mom0.fits
print("PROCESSING: orion_12CO_mom0.fits")
fits_path = '../data/SCIMES/orion_12CO_mom0.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]
    
rms_list = []
inf_list = []
var_list = []
nfl_list = []
nfa_list = []
time_list = []
n_gaussians_list = np.arange(50, 601, 25)
for n_gaussians in n_gaussians_list:
    hdmc = HDMClouds(data, 
                     back_level=1.5, 
                     wcs=wcs, 
                     verbose=False, 
                     n_gaussians=n_gaussians, 
                     eps=100., 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr()
    rmsR, infR, varR, nfa, nfl = hdmc.get_residual_stats(verbose=False)
    rms_list.append(rmsR)
    inf_list.append(infR)
    var_list.append(varR)
    nfa_list.append(nfa)
    nfl_list.append(nfl)
    time_list.append(hdmc.elapsed_time)
    del hdmc
all_results2 = {"rms":rms_list,
               "inf":inf_list,
               "var":var_list,
               "nfa":nfa_list,
               "nfl":nfl_list,
               "time":time_list}

with open('exp-n_gaussians-orionMono.pickle', 'wb') as handle:
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

rms_list = []
inf_list = []
var_list = []
nfl_list = []
nfa_list = []
time_list = []
n_gaussians_list = np.arange(500, 1001, 25)
for n_gaussians in n_gaussians_list:
    hdmc = HDMClouds(data, 
                     back_level=base_level, 
                     freq=spec, 
                     wcs=wcs, 
                     verbose=False, 
                     n_gaussians=n_gaussians, 
                     eps=100., 
                     kappa=2, 
                     gmr_neighbors=64)
    hdmc.build_gmr()
    rmsR, infR, varR, nfa, nfl = hdmc.get_residual_stats(verbose=False)
    rms_list.append(rmsR)
    inf_list.append(infR)
    var_list.append(varR)
    nfa_list.append(nfa)
    nfl_list.append(nfl)
    time_list.append(hdmc.elapsed_time)
    del hdmc
all_results3 = {"rms":rms_list,
               "inf":inf_list,
               "var":var_list,
               "nfa":nfa_list,
               "nfl":nfl_list,
               "time":time_list}

with open('exp-n_gaussians-orionKLCube.pickle', 'wb') as handle:
    pickle.dump(all_results3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()






