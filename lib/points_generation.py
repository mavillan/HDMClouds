import numba
import numpy as np
import numpy.ma as ma
from sklearn.neighbors import NearestNeighbors


def boundary_map(mask):
    m,n = mask.shape
    if isinstance(mask, ma.MaskedArray):
        mask = mask.filled(fill_value=False)
    border_map = np.zeros((m,n), dtype=bool)
    for i in range(m):
        for j in range(n):
            # just verify valid pixels
            if not mask[i,j]: continue
            for p in range(-1,2):
                for q in range(-1,2):
                    if p==q==0: continue
                    if i+p < 0 or j+q < 0: continue
                    if i+p >= m or j+q >= n: continue
                    # in case mask[i,j] has a unusable neighbor pixel
                    # then mask[i,j] is a border pixel
                    if not mask[i+p,j+q]: border_map[i,j] = True
    return border_map


def boundary_map_caa(pixel_map):
    """
    Function specific for CAA boundaries detection.
    """
    m,n = pixel_map.shape
    if isinstance(pixel_map, ma.MaskedArray):
        pixel_map = pixel_map.filled(fill_value=False)
    border_map = np.zeros((m,n), dtype=int)
    for i in range(m):
        for j in range(n):
            # just verify valid pixels
            if pixel_map[i,j]==False: continue
            for p in range(-1,2):
                for q in range(-1,2):
                    if p==q==0: continue
                    if i+p < 0 or j+q < 0: continue
                    if i+p >= m or j+q >= n: continue
                    # in case pixel_map[i,j] has a unusable neighbor pixel
                    # then pixel_map[i,j] is a border pixel
                    if pixel_map[i+p,j+q]!=pixel_map[i,j]: border_map[i,j] = pixel_map[i,j]
    return border_map


def boundary_points_generation(data, mask, neigh_length=3):
    #fixed seed
    np.random.seed(23)
    border_map = boundary_map(mask)

    m,n = data.shape
    prob = np.zeros(border_map.shape)
    prob[border_map] = 1./np.sum(border_map)

    # all points positions
    pix_lenght = 1./max(data.shape)
    _x = (data.shape[0]*pix_lenght) * np.linspace(0., 1., data.shape[0]+1, endpoint=True)
    _y = (data.shape[1]*pix_lenght) * np.linspace(0., 1., data.shape[1]+1, endpoint=True)
    x = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
    y = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
    X,Y  = np.meshgrid(x, y, indexing='ij')
    points_positions = np.vstack( [ X.ravel(), Y.ravel() ]).T
    
    # array with indexes of such centers
    points_indexes = np.arange(0, points_positions.shape[0])
    selected = list()

    while prob.sum()>0:
        # update prob array, to make it sum 1
        prob /= prob.sum()
        try:
            sel = np.random.choice(points_indexes, size=1 , p=prob.ravel(), replace=False)[0]
        except ValueError:
            print("No more points can be selected: {0} boundary points".format(len(sel)))

        # border pixels can't be selected
        index0 = sel // m
        index1 = sel % n
        if index0==0 or index0==m-1 or index1==0 or index1==n-1: continue
        selected.append(sel)

        # update the pixel probabilities array
        nl = neigh_length
        prob[index0-nl:index0+(nl+1), index1-nl:index1+(nl+1)] *= 0.

    return points_positions[selected]


def boundary_points_generation(data, mask, neigh_length=5):
    # all points positions
    pix_lenght = 1./max(data.shape)
    _x = (data.shape[0]*pix_lenght) * np.linspace(0., 1., data.shape[0]+1, endpoint=True)
    _y = (data.shape[1]*pix_lenght) * np.linspace(0., 1., data.shape[1]+1, endpoint=True)
    x = np.asarray( [(_x[i]+_x[i+1])/2 for i in range(len(_x)-1)] )
    y = np.asarray( [(_y[i]+_y[i+1])/2 for i in range(len(_y)-1)] )
    X,Y  = np.meshgrid(x, y, indexing='ij')

    # Obtaining the boundary points through the border_map
    border_map = boundary_map(mask)
    xbound = X[border_map]
    ybound = Y[border_map]
    bound_points = np.vstack([xbound,ybound]).T

    # Nearest neighbor object over the boundary points: 
    # When searching for bound_point, it will reach the point itself + neigh_length other points
    nn = NearestNeighbors(n_neighbors=neigh_length+1, algorithm='ball_tree', n_jobs=-1)
    nn.fit(bound_points)
    nn_indices = nn.kneighbors(bound_points, return_distance=False)

    # array of probabilities (all boundary points have the same prob at start)
    prob = np.ones(bound_points.shape[0])
    prob /= len(bound_points)

    # array with indexes of such centers
    points_indexes = np.arange(0, bound_points.shape[0])
    selected = list()

    # fixed seed for reproducibility
    np.random.seed(23)

    while prob.sum()>0:
        # update prob array, to make it sum 1
        prob /= prob.sum()
        try:
            sel = np.random.choice(points_indexes, size=1 , p=prob, replace=False)[0]
        except ValueError:
            print("No more points can be selected: {0} boundary points".format(len(selected)))
            return np.asarray(selected)

        # nearest neighbors of selected point
        sel_nn_indices = nn_indices[sel]

        # The probability of selected point and its neighbors is set to 0
        prob[sel_nn_indices] = 0.

        # added to the list of selected boundary points
        selected.append(bound_points[sel])

    return np.asarray(selected)


