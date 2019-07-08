import numpy as np

# ignoring annoying/unuseful warnings
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib import gridspec

from fgm_eval  import gm_eval2d_1 as gm_eval
from utils3D import u_eval as u_eval3D
from utils3D import compute_solution
from points_generation import boundary_map_caa

from astropy.visualization import AsymmetricPercentileInterval
import astropy.units as units

font = {'fontname':'Times New Roman'}
umap = {'RA':'RA (J2000)', 'DEC':'Dec (J2000)',
        'GLON':'Galactic Longitude', 'GLAT':'Galactic Latitude',
        'FREQ':'FREQ [GHz]',
        'M/S':'Velocity [M/S]'}

import string
colors = plt.cm.gist_rainbow(np.linspace(0., 1., 100)) 
indexes = np.arange(0,100)
np.random.seed(1);
colors_dict = dict()
for x in string.ascii_uppercase:
    np.random.shuffle(indexes)
    _colors = colors[indexes]
    for i in range(100):
        colors_dict[x+"-"+str(i)] = _colors[i]


def image_plot(data, title=None, cmap=plt.cm.cubehelix, wcs=None,
               vmin=None, vmax=None, unit=None, save_path=None):
    fig = plt.figure(figsize=(12,9))
    if wcs is not None: fig.gca(projection=wcs)
    if vmin is None or vmax is None:
        interval = AsymmetricPercentileInterval(0.35, 99.65, n_samples=100000)
        vmin, vmax = interval.get_limits(data)
        vmin = -0.1*(vmax-vmin) + vmin
        vmax = 0.1*(vmax-vmin) + vmax
    im = plt.imshow(data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
    plt.grid()
    ax = plt.gca()
    ax.invert_yaxis()
    if title is not None: plt.title(title)
    if wcs is not None:
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    if unit is not None: cbar.set_label("[{0}]".format(unit))
    ax.set_aspect('auto')
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=50, bbox_inches='tight')
    plt.show()


def thresholded_image_plot(data, level, cmap=plt.cm.cubehelix, wcs=None,
                           vmin=None, vmax=None):
    if vmin is None or vmax is None:
        interval = AsymmetricPercentileInterval(0.25, 99.75, n_samples=10000)
        vmin, vmax = interval.get_limits(data)
        vmin = -0.1*(vmax-vmin) + vmin
        vmax = 0.1*(vmax-vmin) + vmax
    _data = vmin*np.ones(data.shape)
    mask = data > level
    _data[mask] = data[mask]
    image_plot(_data, title='Data thresholded at: {0}'.format(level), 
               wcs=wcs, vmin=vmin, vmax=vmax)


def solution_plot(data, u, res, title=None, cmap=plt.cm.cubehelix):
    # original data plot
    plt.figure(figsize=(18,12))
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = ax.imshow(data, vmin=data.min(), vmax=data.max(), cmap=plt.cm.cubehelix)
    plt.title('Original data')
    plt.axis('off')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # approximated solution plot
    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(u, vmin=u.min(), vmax=u.max(), cmap=plt.cm.cubehelix)
    plt.title('GM')
    plt.axis('off')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # residual plot
    plt.subplot(1,3,3)
    ax = plt.gca()
    vext = max(np.abs(res.min()), np.abs(res.max()))
    im = ax.imshow(res, vmin=-vext, vmax=vext, cmap=plt.cm.RdBu_r)
    plt.title('Residual')
    plt.axis('off')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def params_plot(c, sig, xc, yc, remove_outlier=False):
    if remove_outlier:
        # just keeping values less than 10 times the mean
        c_med = np.median(c)
        mask1 = c < 10.*c_med
        c = c[mask1]
        xc = xc[mask1]; yc = yc[mask1]
    plt.figure(figsize=(17,7))
    plt.subplot(1,2,1)
    plt.title('Plot of c parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(yc, xc , c=c)
    plt.colorbar()
    plt.subplot(1,2,2)
    if remove_outlier:
        # just keeping values less than 10 times the mean
        sig2_med = np.median(sig**2)
        mask2 = sig**2 < 10.*sig2_med
        sig = sig[mask2]
        xc = xc[mask2]; yc = yc[mask2]
    plt.title('Plot of sig^2 parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(yc, xc, c=sig**2)
    plt.colorbar()
    plt.show()
 

def params_distribution_plot(c, sig, remove_outlier=False):
    if remove_outlier:
        # just keeping values less than 10 times the median
        c_med = np.median(c)
        sig2_med = np.median(sig**2)
        mask1 = c < 10.*c_med
        c = c[mask1]
        mask2 = sig**2 < 10.*sig2_med
        sig = sig[mask2]
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title('C distribution')
    plt.hist(c, bins=10, facecolor='seagreen', edgecolor='black', lw=2)
    plt.subplot(1,2,2)
    plt.title('Sig^2 distribution')
    plt.hist(sig**2, bins=10, facecolor='peru', edgecolor='black', lw=2)
    plt.show()


def residual_plot(residual_variance, residual_entropy, residual_rms, iter_list):
    plt.figure(figsize=(8,5))
    plt.subplot(1,3,1)
    plt.xlim(0, iter_list[-1]+iter_list[0])
    plt.plot(iter_list, residual_rms, 'go-')
    plt.title('Residual RMS')        
    plt.subplot(1,3,2)
    plt.xlim(0, iter_list[-1]+iter_list[0])
    plt.plot(iter_list, residual_variance, 'bo-')
    plt.title('Residual variance')
    plt.subplot(1,3,3)
    plt.xlim(0, iter_list[-1]+iter_list[0])
    plt.plot(iter_list, residual_entropy, 'ro-')
    plt.title('Residual entropy')
    plt.show()


def residual_histogram(residual, title="Histogram of residuals"):
    plt.figure(figsize=(8,4))
    plt.hist(residual, facecolor='seagreen', edgecolor='black', lw=2, bins=20)
    plt.title(title)
    plt.show()
  

def points_plot(data, points=None, color=None, wcs=None, 
                title=None, cmap=plt.cm.cubehelix, save_path=None):
    """
    Function to plot point in the data.

    data : np.array with the data
    points_positions : list with the points to plot
    colors : list of colors used to plot each points
    title : string with title of the plot
    save_path : string with the path where to save the image
    """
    pix_lenght = 1./max(data.shape)
    fig = plt.figure(figsize=(10,10))
    if wcs is not None: fig.gca(projection=wcs)
    # we first plot the data image
    interval = AsymmetricPercentileInterval(0.25, 99.75, n_samples=100000)
    vmin, vmax = interval.get_limits(data)
    vmin = -0.1*(vmax-vmin) + vmin
    vmax = 0.1*(vmax-vmin) + vmax
    im = plt.imshow(data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
    # and then we superimpose the points
    plt.scatter((points[:,1]-pix_lenght/2.)/pix_lenght, (points[:,0]-pix_lenght/2.)/pix_lenght, 
                s=45, facecolor=color, lw=0, alpha=0.95)
    plt.grid()
    ax = plt.gca()
    ax.invert_yaxis()
    if title is not None: plt.title(title)
    if wcs is not None:
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=50, bbox_inches='tight')
    plt.show()


def points_clusters(data, points, labels, hdice_keys, wcs=None, title=None):
    """
    Function to visualize the Isolated Cloud Entities
    """
    pix_lenght = 1./max(data.shape)
    fig = plt.figure(figsize=(12,9))
    if wcs is not None: fig.gca(projection=wcs)
    plt.imshow(data, cmap=plt.cm.gray_r)

    colors = plt.cm.gist_rainbow(np.linspace(0., 1., np.max(labels)+2))
    for i,label in enumerate(range(labels.min(), labels.max()+1)):
        _points = points[labels==label]
        plt.scatter((_points[:,1]-pix_lenght/2.)/pix_lenght, (_points[:,0]-pix_lenght/2.)/pix_lenght, 
                    s=20, facecolor=colors[i], lw=0, alpha=0.9, label="ICE {0}".format(hdice_keys[i]))
    if title is not None: plt.title(title)
    plt.legend(loc="best", prop={'size': 16})  
    plt.grid()
    ax = plt.gca()
    ax.invert_yaxis()
    if wcs is not None:
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
    plt.show()


def ce_plot(hdmc, show_title=False, cmap1=plt.cm.gray_r, 
                 cmap2=plt.cm.gist_rainbow, save_path=None, 
                 wcs=None, unit=None, manual_label=False):

    # first, the original data is plotted in the canvas
    fig = plt.figure(figsize=(12,9))
    if wcs is not None: fig.gca(projection=wcs)
    interval = AsymmetricPercentileInterval(0.25, 99.75, n_samples=100000)
    vmin,vmax = interval.get_limits(hdmc.orig_data)
    vmin = -0.1*(vmax-vmin) + vmin
    vmax = 0.1*(vmax-vmin) + vmax
    ax = plt.gca()
    im = ax.imshow(hdmc.orig_data, cmap=cmap1, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    if wcs is not None:
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
    else:
        plt.tick_params(labelbottom=False, labelleft=False) 
    plt.grid()
    num_ce = len(hdmc.splittable)
    if show_title: plt.title('{0} cloud entities decomposition'.format(num_ce))

    for i,CEid in enumerate(hdmc.splittable):
        ice_key,idx = CEid.split("-")
        hdice = hdmc.hdice_dict[ice_key]

        indexes = hdice.entity_dict[int(idx)]
        params = hdice.get_params_filtered(indexes)
        u = hdice.get_approximation_global(params)
        u = u.reshape(hdmc.shape)

        # to contours are plot at the level of the boundary of each ICE
        levels= [0.5*np.median(hdice.fb[(hdice.fb)>0.])]
        
        if CEid not in colors_dict.keys():
            color = np.array([1.,1.,1.,1.])
        else: color = colors_dict[CEid]
        
        if manual_label:
            try:
                cs = ax.contour(u, levels=levels, colors=[color], linewidths=4)
                ax.clabel(cs, cs.levels, inline=True, fmt=CEid, fontsize=19, manual=manual_label)
            except:
                levels = [u[u>0.].min()]
                cs = ax.contour(u, levels=levels, colors=[color], linewidths=4)
                ax.clabel(cs, cs.levels, inline=True, fmt=CEid, fontsize=19, manual=manual_label)
        else:
            if not levels[0]>u.max():
                cs = ax.contour(u, levels=levels, colors=[color], linewidths=4)
                ax.clabel(cs, cs.levels, inline=True, fmt=CEid, fontsize=19)
                #ax.clabel(cs, cs.levels, inline=True, fmt="S"+CEid.split("-")[1], fontsize=13)
            else:
                _,_,c,sig = params
                levels = [np.mean(u[u>0])]
                print("c_values",c[c>0])
                cs = ax.contour(u, levels=levels, colors=[color], linewidths=4)
                ax.clabel(cs, cs.levels, inline=True, fmt=CEid, fontsize=19)          
        
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=50, bbox_inches='tight')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    if unit is not None: cbar.set_label("[{0}]".format(unit))
    ax.set_aspect('auto')
    plt.show()


def caa_show(data, caa, save_path=None, wcs=None):
    bd_map = boundary_map_caa(caa)
    colors = plt.cm.rainbow(np.linspace(0., 1., caa.max()))
    
    cmap = plt.cm.gray_r
    norm = plt.Normalize(data.min(), data.max())
    rgba = cmap(norm(data))
    
    m,n = data.shape
    for i in range(m):
        for j in range(n):
            if bd_map[i,j]==0: continue
            rgba[i,j,:] = colors[bd_map[i,j]-1]

    patches = []
    for i,color in enumerate(colors):
        colors[bd_map[i,j]-1]
        patches.append(mpatches.Patch(color=color, label='CE {0}'.format(i+1)))

    fig = plt.figure(figsize=(8,8))
    if wcs is not None: fig.gca(projection=wcs)
    im = plt.imshow(rgba)
    plt.grid()
    ax = plt.gca()
    ax.invert_yaxis()
    if wcs is not None:
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
    else:
        plt.tick_params(labelbottom=False, labelleft=False)

    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=150, bbox_inches='tight')
    if wcs is not None:
        cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    ax.set_aspect('auto')
    plt.legend(handles=patches)
    plt.show()

def eccentricity_plot(data, xc, yc, sig, wcs=None):
    # we first compute eccentricity
    ecc = np.empty(len(xc))
    for i,_sig in enumerate(sig.reshape((-1,2))):
        ecc[i] = 1- np.min(_sig)**2/np.max(_sig)**2
        
    x_scale = data.shape[0]-1
    y_scale = data.shape[1]-1
    fig = plt.figure(figsize=(12,9))
    if wcs is not None: fig.gca(projection=wcs)
    ax = plt.gca()
    ax.imshow(data, vmin=data.min(), vmax=data.max(), cmap=plt.cm.gray_r)
    ax.invert_yaxis()
    plt.grid()
    plt.scatter(yc*y_scale, xc*x_scale , c=ecc, s=15, cmap=plt.cm.cool)
    plt.clim(0.,1.)
    plt.colorbar()
    plt.show()  

#################################################################################################
# The following are functions to make equivalent plots for 3D spectroscopic cubes
#################################################################################################

def cube_plot(data, wcs=None, cmap=plt.cm.cubehelix, unit=None, freq=None, save_path=None):
    f = plt.figure(figsize=(16,10))
    if wcs is not None:
        ax = f.add_subplot(221, projection=wcs, slices=("x", 'y', 0))
        ax.coords[2].set_major_formatter('x.x')
        ax.coords[2].set_format_unit(units.GHz)
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
    else: 
        ax = f.add_subplot(221)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.tick_params(labelbottom=False, labelleft=False)
    im = ax.imshow(data.sum(axis=0), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    if unit is not None: cbar.set_label("[{0}]".format(unit))
    ax.invert_yaxis()
    ax.set_aspect('auto')

    if wcs is not None:
        ax = f.add_subplot(223, projection=wcs, slices=("x",0,"y"))
        ax.coords[2].set_major_formatter('x.x')
        ax.coords[2].set_format_unit(units.GHz)
        ax.set_xlabel(umap[wcs.axis_type_names[0]])
        ax.set_ylabel(umap[wcs.axis_type_names[2]])
    else: 
        ax = f.add_subplot(223)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("spectral-axis")
        ax.tick_params(labelbottom=False, labelleft=False)
    im = ax.imshow(data.sum(axis=1), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    if unit is not None: cbar.set_label("[{0}]".format(unit))
    ax.set_aspect('auto')

    if wcs is not None:
        ax = f.add_subplot(222, projection=wcs, slices=(0,"y","x"))
        ax.coords[2].set_major_formatter('x.x')
        ax.coords[2].set_format_unit(units.GHz)
        ax.set_ylabel(umap[wcs.axis_type_names[1]])
        ax.set_xlabel(umap[wcs.axis_type_names[2]])
    else: 
        ax = f.add_subplot(222)
        ax.set_xlabel("spectral-axis")
        ax.set_ylabel("y-axis")
        ax.tick_params(labelbottom=False, labelleft=False)
    im = ax.imshow(data.sum(axis=2).T, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    if unit is not None: cbar.set_label("[{0}]".format(unit))
    ax.invert_yaxis()
    ax.set_aspect('auto')

    if freq is not None:
        ax = f.add_subplot(224)
        flux = data.sum(axis=(1,2))/data.sum()
        ax.plot(freq, flux, 'o--', lw=1, color="red", ms=6)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if wcs is not None: ax.set_xlabel("FREQ [GHz]")
        else: 
            ax.set_xlabel("spectral-axis")
            ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_ylabel("Standardized flux")
        ax.grid()
        ax.set_aspect('auto')
    plt.show()
    
    
def points_plot3d(points, title=None, color="red"):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    # visualization of points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color, marker='o', s=7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show()


def slices_plot(data, slc):
    plt.figure(figsize=(8,8))
    im = plt.imshow(data[slc], vmin=0, vmax=1., cmap=plt.cm.cubehelix)
    plt.title('3D cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
  

def solution_plot_3d(data, u, cmap=plt.cm.cubehelix):
    plt.figure(figsize=(12,16))
    # original data plot
    ax = plt.subplot(3,2,1)
    im = ax.imshow(data.sum(axis=0), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    #ax.set_aspect('auto')

    ax = plt.subplot(3,2,3)
    im = ax.imshow(data.sum(axis=1), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    #ax.set_aspect('auto')

    plt.subplot(3,2,5)
    im = ax.imshow(data.sum(axis=2), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    #ax.set_aspect('auto')  

    # gaussian mixture plot
    plt.subplot(3,2,2)
    im = ax.imshow(u.sum(axis=0), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    #ax.set_aspect('auto')

    plt.subplot(3,2,4)
    im = ax.imshow(u.sum(axis=1), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    #ax.set_aspect('auto')

    plt.subplot(3,2,6)
    im = ax.imshow(u.sum(axis=2), cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    #ax.set_aspect('auto') 
    plt.show()


def comparative_slices_plot(data1, data2, slc):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    im = plt.imshow(data1[slc], vmin=0, vmax=1., cmap=plt.cm.cubehelix)
    plt.title('3D original cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1,2,2)
    im = plt.imshow(data2[slc], vmin=0, vmax=1., cmap=plt.cm.cubehelix)
    plt.title('3D approximated cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def ce_plot_3d(hdmc, show_title=False, cmap1=plt.cm.cubehelix, 
                 cmap2=plt.cm.gist_rainbow, save_path=None,
                 wcs=None, unit=None, manual_label=False):
    # generating the color for the cloud entities
    fig = plt.figure(figsize=(16,10))
    vmin = hdmc.vmin
    vmax = hdmc.vmax
    num_ce = len(hdmc.splittable)
    color = cmap2(np.linspace(0., 1., num_ce))
    if show_title: plt.title('{0} cloud entities decomposition'.format(num_ce))

    for axis in range(3):
        if axis==0:
            if wcs is not None:
                ax = plt.subplot(221, projection=wcs, slices=("x", "y", 0))
                ax.set_xlabel("RA (J2000)")
                ax.set_ylabel("Dec (J2000)")
                ax.coords[2].set_major_formatter('x.x')
                ax.coords[2].set_format_unit(units.GHz)
            else: 
                ax = plt.subplot(221)
                ax.set_xlabel("x-axis")
                ax.set_ylabel("y-axis")
                ax.tick_params(labelbottom=False, labelleft=False)
        elif axis==1:
            if wcs is not None:
                ax = plt.subplot(223, projection=wcs, slices=("x",0,"y"))
                ax.set_xlabel("RA (J2000)")
                ax.set_ylabel("FREQ [GHz]")
                ax.coords[2].set_major_formatter('x.x')
                ax.coords[2].set_format_unit(units.GHz)
            else: 
                ax = plt.subplot(223)
                ax.set_xlabel("x-axis")
                ax.set_ylabel("spectral-axis")
                ax.tick_params(labelbottom=False, labelleft=False)
        elif axis==2:
            if wcs is not None:
                ax = plt.subplot(222, projection=wcs, slices=(0,"y","x"))
                ax.set_xlabel("FREQ [GHz]")
                ax.set_ylabel("Dec (J2000)")
                ax.locator_params(axis="x", nbins=2)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.coords[2].set_major_formatter('x.x')
                ax.coords[2].set_format_unit(units.GHz)
            else:
                ax = plt.subplot(222)
                ax.set_xlabel("spectral-axis")
                ax.set_ylabel("y-axis")
                ax.tick_params(labelbottom=False, labelleft=False)

        # stacked data, mapping to [0,1] and display
        _data = hdmc.orig_data.sum(axis=axis)
        if axis==2: _data = _data.T
        im = ax.imshow(_data, cmap=cmap1)
        cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
        cbar.ax.tick_params(labelsize=19)
        if axis==0 or axis==2: ax.invert_yaxis()

        for i,CEid in enumerate(hdmc.splittable):
            ice_key,idx = CEid.split("-")
            hdice = hdmc.hdice_dict[ice_key]

            indexes = hdice.entity_dict[int(idx)]
            params = hdice.get_params_filtered(indexes)
            u = hdice.get_approximation_global(params)
            u *= (vmax-vmin); u += vmin; u += hdmc.back_level
            u = u.reshape(hdmc.shape)
            u[u<hdmc.back_level] = 0
            _u = u.sum(axis=axis)
            if axis==2: _u = _u.T  
                
            levels = [np.percentile(_u[_u>0], 5)]
            cs = ax.contour(_u, levels=levels, colors=[color[i]], linewidths=4)
            ax.clabel(cs, cs.levels, inline=True, fmt=CEid, fontsize=24)
        ax.set_aspect('auto')
    
    ax = plt.subplot(224)
    total_flux = hdmc.orig_data.sum()
    for i,CEid in enumerate(hdmc.splittable):
        ice_key,idx = CEid.split("-")
        hdice = hdmc.hdice_dict[ice_key]

        indexes = hdice.entity_dict[int(idx)]
        params = hdice.get_params_filtered(indexes)
        u = hdice.get_approximation_global(params)
        u *= (vmax-vmin); u += vmin; u += hdmc.back_level
        u = u.reshape(hdmc.shape)
        f = u.sum(axis=(1,2))
        f /= total_flux
            
        if hdmc.freq is not None:
            ax.plot(hdmc.freq, f, '--', lw=4, color=color[i], ms=6)
            ax.text(hdmc.freq[np.argmax(f)], np.max(f), CEid, color=color[i], fontsize=24)
            ax.locator_params(nbins=5, axis='x')
        else:
            ax.plot(f, '--', lw=4, color=color[i], ms=6)
    if wcs is not None: ax.set_xlabel("FREQ [GHz]")
    else: 
        ax.set_xlabel("spectral-axis")
        ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_ylabel("Standardized flux")
    ax.grid() 
    ax.set_aspect('auto')
       
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=150, bbox_inches='tight')
    plt.show()


#################################################################################################
# General purpose functions
#################################################################################################

def stat_plots(x_var, y_list, labels, xlabel=None, ylabel=None, save_name=None, legend=False):
    """
    Function to plot a single residual stat for multiple images
    """
    plt.figure(figsize=(7,4))
    colors = plt.cm.rainbow(np.linspace(0., 1., 100))
    np.random.seed(0)
    np.random.shuffle(colors)
    colors = colors[0:len(y_list)]
    for i,y_var in enumerate(y_list):
        plt.plot(x_var, y_var, c=colors[i], marker='o', label=labels[i])
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    plt.grid()
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()
    if legend: plt.legend(loc='best', prop={'size':20})
    if save_name is not None:
        plt.savefig(save_name, format='eps', dpi=1000, bbox_inches='tight')
    plt.show()


def stat_plots_log(x_var, y_list, labels, xlabel=None, ylabel=None, save_name=None, legend=False):
    """
    Function to plot a single residual stat for multiple images
    """
    plt.figure(figsize=(14,4))
    colors = plt.cm.rainbow(np.linspace(0., 1., 100))
    np.random.seed(0)
    np.random.shuffle(colors)
    colors = colors[0:len(y_list)]
    plt.subplot(1,2,1)
    for i,y_var in enumerate(y_list):
        plt.plot(x_var, y_var, c=colors[i], marker='o', label=labels[i])
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    plt.grid()
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()
    if legend: plt.legend(loc='best', prop={'size':20})

    plt.subplot(1,2,2)
    for i,y_var in enumerate(y_list):
        plt.loglog(x_var, y_var, c=colors[i], marker='o', label=labels[i])
    y_ref = 0.01 * x_var**2
    plt.loglog(x_var, y_ref, c='red')
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    plt.grid()
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, format='eps', dpi=1000, bbox_inches='tight')
    plt.show()


def all_stats_plot(x_var, r_stats, x_label='', loglog=False, n=5, slope=None, name=None):
    """
    Function to plot all the residual stats for a single image
    """
    y_label = ['RMS', 'Flux addition', 'Flux lost', 'Sharpness']
    # unpacking the values
    r_stats_list = []
    r_stats_list.append( np.array([rms for (_,_,rms,_,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([flux for (_,_,_,flux,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([flux for (_,_,_,_,flux,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([sharp for (_,_,_,_,_,_,_,sharp,_) in r_stats]) )

    colors = plt.cm.rainbow(np.linspace(0., 1., len(r_stats_list)))
    fig = plt.figure(figsize=(15,7))
    m = r_stats_list[0].max(); m=1
    plt.plot(x_var, r_stats_list[0]/m, color=colors[0], marker='o', label='RMS x {0:.3f}'.format(m))
    m = r_stats_list[1].max(); m=1
    plt.plot(x_var, r_stats_list[1]/m, color=colors[1], marker='o', label='Flux addition x {0:.3f}'.format(m))
    m = r_stats_list[2].max(); m=1
    plt.plot(x_var, r_stats_list[2]/m, color=colors[2], marker='o', label='Flux lost x {0:.3f}'.format(m))
    m = r_stats_list[3].max()
    plt.grid()
    plt.tick_params(axis='both', which='major')
    plt.xlabel(x_label)
    plt.legend(loc='best', prop={'size':20})

    if name is not None: plt.savefig(name, format='eps', dpi=1000, bbox_inches='tight')
    plt.show()
