import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import square, disk, cube, ball
from ipywidgets import interact, fixed, FloatSlider
from astropy.visualization import AsymmetricPercentileInterval

from utils import estimate_rms

import warnings
warnings.filterwarnings("ignore", message="Only one label was provided to `remove_small_objects`.")

umap = {'RA':'RA (J2000)', 'DEC':'Dec (J2000)',
        'GLON':'Galactic Longitude', 'GLAT':'Galactic Latitude'}
        

def compute_mask(data, back_level, min_obj_size=20, min_hole_size=10):
    # thresholding
    mask = data>=back_level
    if data.ndim==2:
        # structuring elements
        disk1 = disk(1)
        square2 = square(2)
        # cleaning
        # opening: remove small bright spots and connect small dark cracks
        mask = binary_opening(mask, selem=disk1)
        mask = remove_small_objects(mask.astype(bool), min_size=min_obj_size)
        # closing: remove small dark spots and connect small bright cracks
        mask = binary_closing(mask, selem=square2)
        mask = remove_small_holes(mask, min_size=min_hole_size)
    if data.ndim==3:
        # structuring elements
        ball1 = ball(1)
        cube2 = cube(2)
        # cleaning
        # opening: remove small bright spots and connect small dark cracks
        mask = binary_opening(mask, selem=ball1)
        mask = remove_small_objects(mask, min_size=min_obj_size)
        # closing: remove small dark spots and connect small bright cracks
        mask = binary_closing(mask, selem=cube2)
        mask = remove_small_holes(mask, min_size=min_hole_size)
    return mask


def mask_show_helper(data, back_level, wcs=None):
    mask = compute_mask(data, back_level)

    # visualization of original image
    plt.figure(figsize=(20,6))
    ax1 = plt.subplot(131, projection=wcs)
    plt.title("Original Image")
    interval = AsymmetricPercentileInterval(0.25, 99.75, n_samples=100000)
    vmin, vmax = interval.get_limits(data)
    vmin = -0.1*(vmax-vmin) + vmin
    vmax = 0.1*(vmax-vmin) + vmax
    im = plt.imshow(data, cmap=plt.cm.cubehelix, interpolation=None, vmin=vmin, vmax=vmax)
    plt.grid()
    ax1.invert_yaxis()
    if wcs is not None:
        plt.xlabel(umap[wcs.axis_type_names[0]])
        plt.ylabel(umap[wcs.axis_type_names[1]])
    plt.colorbar(im, ax=ax1, pad=0.01, aspect=30)
    ax1.set_aspect('auto')

    # visualization of mask of significant emission pixels
    ax2 = plt.subplot(132, projection=wcs)
    plt.title("Mask")
    im = plt.imshow(mask, cmap=plt.cm.cubehelix, vmin=0, vmax=1)
    plt.grid()
    ax2.invert_yaxis()
    if wcs is not None:
        plt.xlabel(umap[wcs.axis_type_names[0]])
    plt.colorbar(im, ax=ax2, pad=0.01, aspect=30)
    ax2.set_aspect('auto')

    # visualization of the masked image
    _data = np.copy(data)
    _data[~mask] = 0
    ax3 = plt.subplot(133, projection=wcs)
    plt.title("Masked Image")
    im = plt.imshow(_data, cmap=plt.cm.cubehelix, interpolation=None, vmin=vmin, vmax=vmax)
    plt.grid()
    ax3.invert_yaxis()
    if wcs is not None:
        plt.xlabel(umap[wcs.axis_type_names[0]])
    plt.colorbar(im, ax=ax3, pad=0.01, aspect=30)
    ax3.set_aspect('auto')
    plt.show()

    # storing last  values used
    print("back_level: ",back_level)


def preprocessing(data, wcs=None): 
    # first estimation of the background level
    back_level = estimate_rms(data)
    minv = 0.05*back_level
    maxv = 2*back_level
    step = (maxv-minv)/100.
    # interactive selection of back_level and mask
    interact(mask_show_helper, 
             data=fixed(data), 
             back_level=FloatSlider(min=minv,max=maxv,step=step,value=back_level,readout_format='.4f'), 
                                    wcs=fixed(wcs));

