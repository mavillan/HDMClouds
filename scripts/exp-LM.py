%matplotlib inline

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
matplotlib.rcParams.update({'font.size': 20})



# case 1: Orion.cont.image.fits
print("PROCESSING: Orion.cont.image.fits")
fits_path = '../data/images/Orion.cont.image.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]

times = []
rms_errors = []
inf_errors = []
hdmc_ig = HDMClouds(data, back_level=0.089, wcs=wcs, verbose=False, n_gaussians=200, eps=100, kappa=1, gmr_neighbors=64)
for max_nfev in range(100, 10001, 100):
    print("MAX_NFEV:",max_nfev)
    hdmc = copy.deepcopy(hdmc_ig)
    hdmc.build_gmr(max_nfev=max_nfev)
    times.append(hdmc.elapsed_time)
    rms,inf,_,_,_ = hdmc.get_residual_stats(verbose=False)
    rms_errors.append(rms)
    inf_errors.append(inf)
    del hdmc
orionKLresults = {"times":times, "rms":rms_errors, "inf":inf_errors}
with open('exp-LM-orionKL.pickle', 'wb') as handle:
    pickle.dump(orionKLresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()



# case 2: orion_12CO_mom0.fits
print("PROCESSING: orion_12CO_mom0.fits")
fits_path = '../data/SCIMES/orion_12CO_mom0.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]

times1 = []
rms_errors1 = []
inf_errors1 = []
hdmc_ig = HDMClouds(data, back_level=1.5, wcs=wcs, verbose=False, n_gaussians=400, eps=100, kappa=1, gmr_neighbors=64)
for max_nfev in range(100, 10001, 100):
    print("MAX_NFEV:",max_nfev)
    hdmc = copy.deepcopy(hdmc_ig)
    hdmc.build_gmr(max_nfev=max_nfev)
    times1.append(hdmc.elapsed_time)
    rms,inf,_,_,_ = hdmc.get_residual_stats(verbose=False)
    rms_errors1.append(rms)
    inf_errors1.append(inf)
    del hdmc
orionMonoresults = {"times":times1, "rms":rms_errors1, "inf":inf_errors1}
with open('exp-LM-orionMono.pickle', 'wb') as handle:
    pickle.dump(orionMonoresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()



# case 3: Orion.methanol.cbc.contsub.image.fits
print("PROCESSING: Orion.methanol.cbc.contsub.image.fits")
fits_path = '../data/cubes/Orion.methanol.cbc.contsub.image.fits'
loaded_fits = load_data(fits_path)
data = loaded_fits["data"]
hdu = loaded_fits["hdu"]
wcs = loaded_fits["wcs"]
spec = loaded_fits["spec"]

times2 = []
rms_errors2 = []
inf_errors2 = []
base_level = estimate_rms(data)
hdmc_ig = HDMClouds(data, back_level=back_level, freq=spec, wcs=wcs, verbose=False, n_gaussians=750, eps=100, kappa=1, gmr_neighbors=64)
for max_nfev in range(1000, 100001, 1000):
    print("MAX_NFEV:",max_nfev)
    hdmc = copy.deepcopy(hdmc_ig)
    hdmc.build_gmr(max_nfev=max_nfev)
    times2.append(hdmc.elapsed_time)
    rms,inf,_,_,_ = hdmc.get_residual_stats(verbose=False)
    rms_errors2.append(rms)
    inf_errors2.append(inf)
    del hdmc
orionKLCuberesults = {"times":times2, "rms":rms_errors2, "inf":inf_errors2}
with open('exp-LM-orionKLCube.pickle', 'wb') as handle:
    pickle.dump(orionKLCuberesults, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

