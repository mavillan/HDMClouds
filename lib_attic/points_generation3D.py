import ghalton
import copy
import numpy as np
import scipy.stats as st


def _inv_gaussian_kernel(kernlen=3, sig=0.1):
    """
    Returns a 2D Gaussian kernel array.
    """
    interval = (2*sig+1.)/(kernlen)
    x = np.linspace(-sig-interval/2., sig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel.max()-kernel


def boundary_generation(n_boundary):
    xb = []
    yb = []

    for val in np.linspace(0., 1., n_boundary+1)[0:-1]:
        xb.append(val)
        yb.append(0.)
    for val in np.linspace(0., 1., n_boundary+1)[0:-1]:
        xb.append(1.)
        yb.append(val)
    for val in np.linspace(0., 1., n_boundary+1)[::-1][:-1]:
        xb.append(val)
        yb.append(1.)
    for val in np.linspace(0., 1., n_boundary+1)[::-1][:-1]:
        xb.append(0.)
        yb.append(val)
    xb = np.asarray(xb)
    yb = np.asarray(yb)
    boundary_points = np.vstack([xb,yb]).T
    return boundary_points


def random_centers_generation(data, n_centers, base_level=None, power=5.):
    # fixed seed
    np.random.seed(0)
    
    data = np.copy(data)
    
    # unusable pixels mask
    if base_level is not None:
        mask = data <= base_level
        if np.sum(~mask) < n_centers:
            print('The number of usable pixels is less than n_centers')
            return None
        
    # applying power and re-normalizing
    data **= power
    data /= data.max()

    # data dimensions
    l,m,n = data.shape
    
    # center points positions
    x = np.linspace(0., 1., l+2, endpoint=True)[1:-1]
    y = np.linspace(0., 1., m+2, endpoint=True)[1:-1]
    z = np.linspace(0., 1., n+2, endpoint=True)[1:-1]
    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
    points_positions = np.vstack( [ X.ravel(), Y.ravel(), Z.ravel() ]).T
    
    # array with indexes of such centers
    points_indexes = np.arange(0, points_positions.shape[0], dtype=int)
    
    # array with probabilities of selection for each center
    if isinstance(mask, np.ndarray):
        data[mask] = 0.
        prob = data/data.sum()
    else:
        prob = data/data.sum()
    
    # convolution kernel
    #K = np.array([[0.5, 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.5]])
    #K = _inv_gaussian_kernel(kernlen=3, sig=3.)
    
    selected = []
    while len(selected)!=n_centers:
        sel = np.random.choice(points_indexes, size=1 , p=prob.ravel(), replace=False)[0]
        # border pixels can't be selected
        index0 = sel / (m*n)
        index1 = (sel/ n) % m 
        index2 = sel % n
        if index0==0 or index0==l-1 or index1==0 or index1==m-1 or index2==0 or index2==n-1: continue
        selected.append(sel)
        # update the pixel probabilities array
        #prob[index0-1:index0+2, index1-1:index1+2] *= K
        prob[index0-1:index0+2, index1-1:index1+2, index2-1:index2+2] *= 0.5
        prob[index0, index1, index2] *= 0.
        prob /= prob.sum()
        
    return points_positions[selected]


def qrandom_centers_generation(dfunc, n_centers, base_level, ndim=2, get_size=50):
    # generating the sequencer
    sequencer = ghalton.Halton(ndim)

    points_positions = []
    n_selected = 0

    while True:
        points = np.asarray(sequencer.get(get_size))
        values = dfunc(points)

        for i in range(get_size):
            if values[i] > base_level:
                points_positions.append(points[i])
                n_selected += 1
            if n_selected == n_centers:
                return np.asarray(points_positions)