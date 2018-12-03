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
matplotlib.rcParams.update({'font.size': 20})

if len(sys.argv)>1:
    out_dir = sys.argv[1]
    if out_dir[-1]!="/": out_dir+="/"
else: out_dir = "./"

# case 1: Orion.cont.image.fits
if not os.path.isfile(out_dir+'exp-LM-orionKL.pickle'):
    print("PROCESSING: Orion.cont.image.fits")
    fits_path = '/user/m/marvill/HDMClouds/data/images/Orion.cont.image.fits'
    loaded_fits = load_data(fits_path)
    data = loaded_fits["data"]
    hdu = loaded_fits["hdu"]
    wcs = loaded_fits["wcs"]
    
    times = []
    rms_errors = []; inf_errors = []; var_errors = []
    _rms_errors = []; _inf_errors = []; _var_errors = []
    hdmc_ig = HDMClouds(data, 
                        back_level=0.089, 
                        wcs=wcs, 
                        verbose=False, 
                        n_gaussians=250, 
                        eps=100, 
                        kappa=1.5, 
                        gmr_neighbors=64)
    for max_nfev in range(100, 10001, 100):
        print("MAX_NFEV:",max_nfev)
        hdmc = copy.deepcopy(hdmc_ig)
        hdmc.build_gmr(max_nfev=max_nfev)
        times.append(hdmc.elapsed_time)
        # residuals on evaluation points
        _rms,_inf,_var,_,_ = hdmc._get_residual_stats()
        _rms_errors.append(_rms)
        _inf_errors.append(_inf)
        _var_errors.append(_var)
        # residual on grid points
        rms,inf,var,_,_ = hdmc.get_residual_stats()
        rms_errors.append(rms)
        inf_errors.append(inf)
        var_errors.append(var)
        del hdmc
    orionKLresults = {"times":times, "rms":rms_errors, "inf":inf_errors, "var":var_errors,
                      "_rms":_rms_errors, "_inf":_inf_errors, "_var":_var_errors}
    with open(out_dir+'exp-LM-orionKL.pickle', 'wb') as handle:
        pickle.dump(orionKLresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()



# case 2: orion_12CO_mom0.fits
if not os.path.isfile(out_dir+'exp-LM-orionMono.pickle'):
    print("PROCESSING: orion_12CO_mom0.fits")
    fits_path = '/user/m/marvill/HDMClouds/data/SCIMES/orion_12CO_mom0.fits'
    loaded_fits = load_data(fits_path)
    data = loaded_fits["data"]
    hdu = loaded_fits["hdu"]
    wcs = loaded_fits["wcs"]

    times = []
    rms_errors = []; inf_errors = []; var_errors = []
    _rms_errors = []; _inf_errors = []; _var_errors = []
    hdmc_ig = HDMClouds(data, 
                        back_level=1.5, 
                        wcs=wcs, 
                        verbose=False, 
                        n_gaussians=400, 
                        eps=100, 
                        kappa=1.5, 
                        gmr_neighbors=64)
    for max_nfev in range(100, 10001, 100):
        print("MAX_NFEV:",max_nfev)
        hdmc = copy.deepcopy(hdmc_ig)
        hdmc.build_gmr(max_nfev=max_nfev)
        times.append(hdmc.elapsed_time)
         # residuals on evaluation points
        _rms,_inf,_var,_,_ = hdmc._get_residual_stats()
        _rms_errors.append(_rms)
        _inf_errors.append(_inf)
        _var_errors.append(_var)
        # residual on grid points
        rms,inf,var,_,_ = hdmc.get_residual_stats()
        rms_errors.append(rms)
        inf_errors.append(inf)
        var_errors.append(var)
        del hdmc
    orionMonoresults = {"times":times, "rms":rms_errors, "inf":inf_errors, "var":var_errors,
                        "_rms":_rms_errors, "_inf":_inf_errors, "_var":_var_errors}
    with open(out_dir+'exp-LM-orionMono.pickle', 'wb') as handle:
        pickle.dump(orionMonoresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()



# case 3: Orion.methanol.cbc.contsub.image.fits
if not os.path.isfile(out_dir+'exp-LM-orionKLCube.pickle'):
    print("PROCESSING: Orion.methanol.cbc.contsub.image.fits")
    fits_path = '/user/m/marvill/HDMClouds/data/cubes/Orion.methanol.cbc.contsub.image.fits'
    loaded_fits = load_data(fits_path)
    data = loaded_fits["data"]
    hdu = loaded_fits["hdu"]
    wcs = loaded_fits["wcs"]
    spec = loaded_fits["spec"]

    times = []
    rms_errors = []; inf_errors = []; var_errors = []
    _rms_errors = []; _inf_errors = []; _var_errors = []
    back_level = estimate_rms(data)
    hdmc_ig = HDMClouds(data, 
                        back_level=back_level, 
                        freq=spec, 
                        wcs=wcs, 
                        verbose=False, 
                        n_gaussians=800, 
                        eps=100, 
                        kappa=1.5, 
                        gmr_neighbors=64)
    for max_nfev in range(1000, 50001, 2000):
        print("MAX_NFEV:",max_nfev)
        hdmc = copy.deepcopy(hdmc_ig)
        hdmc.build_gmr(max_nfev=max_nfev)
        times.append(hdmc.elapsed_time)
         # residuals on evaluation points
        _rms,_inf,_var,_,_ = hdmc._get_residual_stats()
        _rms_errors.append(_rms)
        _inf_errors.append(_inf)
        _var_errors.append(_var)
        # residual on grid points
        rms,inf,var,_,_ = hdmc.get_residual_stats()
        rms_errors.append(rms)
        inf_errors.append(inf)
        var_errors.append(var)
        del hdmc
    orionKLCuberesults = {"times":times, "rms":rms_errors, "inf":inf_errors, "var":var_errors,
                          "_rms":_rms_errors, "_inf":_inf_errors, "_var":_var_errors}
    with open(out_dir+'exp-LM-orionKLCube.pickle', 'wb') as handle:
        pickle.dump(orionKLCuberesults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

