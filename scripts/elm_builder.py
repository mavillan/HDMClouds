import pickle
import argparse
import numpy as np
import sys
# VarClump functions
sys.path.append('/user/m/marvill/VarClump/lib/')
from utils3D import *
#from graph import *
from points_generation3D import *
from initial_guess3D import *
from variational3D import *

data_path = '/user/m/marvill/VarClump/data/cubes/'

if __name__ == '__main__':
    # input parameters parsing
    parser = argparse.ArgumentParser(description='ELM instances builder script')
    #parser.add_argument('fits_path', help='Path to FITS file', default=None)
    parser.add_argument('n_center', help='Number of center points to use', type=int)
    #parser.add_argument('points_method', help='Points generation method')
    #parser.add_argument('solver_method', help='Method used in elm_solver() function')
    args = parser.parse_args()

    # LOADING DATA
    #fits_path = data_path+'Antennae_North.CO2_1Line.Clean.image.fits'
    fits_path = data_path+'Orion.methanol.cbc.contsub.image.fits'
    x, y, z, data, dfunc = load_data(fits_path)

    # base level over which usable pixels will be taken
    base_level = 1.5*estimate_rms(data)

    # number of points of each type
    Nb = 0
    Nc = args.n_center
    Ne = 5*Nc-4*Nb

    # if args.points_method == 'standard':
    #     points = qrandom_centers_generation(dfunc, Ne, base_level, ndim=3)
    #     center_points = points[0:Nc]
    #     collocation_points = points[0:Ne]
    #     boundary_points = boundary_generation(Nb)
    # elif args.points_method == 'random':
    #     center_points = random_centers_generation(data, Nc, base_level=base_level, power=3.)
    #     collocation_points = qrandom_centers_generation(dfunc, Ne, base_level, ndim=3)
    #     boundary_points = boundary_generation(Nb)
    # elif args.points_method == 'halton1':
    #     collocation_points = qrandom_centers_generation(dfunc, Ne, base_level, ndim=3)
    #     center_points = collocation_points[0:Nc]
    #     boundary_points = boundary_generation(Nb)
    # elif args.points_method == 'halton2':
    #     points = qrandom_centers_generation(dfunc, Nc+Ne, base_level, ndim=3)
    #     center_points = points[0:Nc]
    #     collocation_points = points[Nc:Nc+Ne]
    #     boundary_points = boundary_generation(Nb)

    points = qrandom_centers_generation(dfunc, Ne, base_level, ndim=3)
    center_points = points[0:Nc]
    collocation_points = points[0:Ne]

    # right format
    xc = center_points[:,0]
    yc = center_points[:,1]
    zc = center_points[:,2]
    xe = collocation_points[:,0]
    ye = collocation_points[:,1]
    ze = collocation_points[:,2]
    xb = np.empty(Nb)
    yb = np.empty(Nb)
    zb = np.empty(Nb)

    # initial parameters estimation
    minsig = 5*( (1./data.shape[0] + 1./data.shape[1] + 1./data.shape[2])/3. ) / 6.
    maxsig = 30*minsig
    dist_matrix = build_dist_matrix(center_points) 
    c0, sig0 = estimate_initial_guess(center_points, dist_matrix, dfunc, minsig=minsig, maxsig=maxsig, method='min_dist')

    # ELModel instantiation
    elm = ELModel(data, dfunc, data.shape, xe, ye, ze, xc, yc, zc, xb, yb, zb, c0, sig0, 
                  a=1., lamb1=1, base_level=base_level, pix_freedom=3., minsig=minsig, maxsig=maxsig)

    # solving it
    nfev =  100 * ( len(elm.get_params())+1 ) 
    elm_solver(elm, method='standard', max_nfev=nfev, verbose=False)


    # storing results
    out = open( '/user/m/marvill/VarClump/results/elm-{0}C.pkl'.format(Nc), 'wb' )
    pickle.dump(elm, out, protocol=2)
    out.close()

