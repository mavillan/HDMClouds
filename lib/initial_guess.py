import numpy as np
from utils import build_dist_matrix


def estimate_initial_guess(center_points, dfunc, R=0.05, minsig=None, maxsig=None, method='min_dist'):
    # building the distance matrix
    dist_matrix = build_dist_matrix(center_points)
    
    m = center_points.shape[0]
    c_arr = np.empty(m, dtype=float)
    sig_arr = np.empty(m, dtype=float)
    
    if method=='mean_dist':
        f = 1./np.sqrt(np.log(2.))
        mean_dist = np.zeros(m, dtype=float)
        num_neigh = np.zeros(m, dtype=float)   
        for i in range(m):
            for j in range(m):
                #dont take into account the same point
                if i==j: continue
                d = dist_matrix[i,j]
                #dont take into account points outside R radius
                if d>R: continue
                num_neigh[i] += 1
                mean_dist[i] += d
            """
            Key Idea: The mean distance to neighbors acurrs when the
            gaussian function has decayed to the half
            """
            if num_neigh[i]==0:
                c_arr[i] = dfunc(center_points[i])[0]
                sig_arr[i] = minsig
            else:
                mean_dist[i] /= num_neigh[i]
                c_arr[i] = dfunc(center_points[i])[0]/num_neigh[i]
                #c_arr[i] = dfunc(*center_points[i])[0]*mean_dist[i]**2
                sig_arr[i] = f*mean_dist[i]
                
    elif method=='min_dist':
        min_dist = np.inf*np.ones(m, dtype=float)
        num_neigh = np.zeros(m, dtype=float)
        #first we find the distance to the nearest neighbor
        for i in range(m):
            for j in range(m):
                #dont take into account the same point or
                #center point in the same position (anomalous case)
                if dist_matrix[i,j] == 0.: continue
                if dist_matrix[i,j] < min_dist[i]:
                    min_dist[i] = dist_matrix[i,j]
        #second, we find the number of neighbors on the neighborhood
        for i in range(m):
            for j in range(m):
                #dont take into account the same point
                if i==j: continue
                d = dist_matrix[i,j]
                if d > 3*min_dist[i]: continue
                num_neigh[i] += 1
            """
            some explanation here
            """
            if num_neigh[i]==0:
                c_arr[i] = 0.5*dfunc(center_points[i])[0]
                sig_arr[i] = minsig
            else:
                c_arr[i] = 0.5*dfunc(center_points[i])[0]/(num_neigh[i]+1)
                if min_dist[i] < minsig:
                    sig_arr[i] = minsig
                elif min_dist[i] > maxsig:
                    sig_arr[i] = 0.99*maxsig
                else:
                    sig_arr[i] = min_dist[i]
    return (c_arr,sig_arr)