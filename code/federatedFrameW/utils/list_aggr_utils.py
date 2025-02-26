import numpy as np
import random

def ra_FedAvg(updates, weights, **kwargs):
    return [np.average(updates, weights=weights, axis=0)]

def ra_inner_Cent1P(updates, weights, **kwargs):
    """
    1-center clustering(CenterwO, inner): https://arxiv.org/abs/2312.12835
    Algorithm 1 of Shenmaier 2015 https://link.springer.com/chapter/10.1007/978-88-7642-475-5_92
    For every point x in updates, find the closest k=eta*n points to it, 
    denote the distance between the k-th nearest point to x by radii, choose the ball with smallest radii,
    then return the weighted average of inner points.
    """
    eta = kwargs['in_eta']
    n = len(updates)
    k = int(np.floor(eta*n))
    radii = [None]*n

    for i in range(n):
        _, radii[i] = k_closest(updates, updates[i], k)

    min_r_index = np.argmin(radii)
    k_data_idx,_ = k_closest(updates, updates[min_r_index], k)
    # if min_r_index not in k_data_idx:  # verifed OK
    #     raise RuntimeError('ra_inner_Cent1P, min_r_index not in k_data_idx!') 

    k_data = [updates[i] for i in k_data_idx]
    k_data_weights = [weights[i] for i in k_data_idx]
    inner_center = np.average(k_data, weights=k_data_weights, axis=0)
    
    return [inner_center]

def ra_outer_Cent1P(updates, weights, **kwargs):
    """
    same as ra_inner_Cent1P, except return the weighted average of outer points.
    """
    eta = kwargs['out_eta']
    n = len(updates)
    k = int(np.floor(eta*n))
    radii = [None]*n

    for i in range(n):
        _, radii[i] = k_closest(updates, updates[i], k)

    min_r_index = np.argmin(radii)
    k_data_idx,_ = k_closest(updates, updates[min_r_index], k)

    k_data = [updates[i] for i in range(n) if i not in k_data_idx]
    k_data_weights = [weights[i] for i in range(n) if i not in k_data_idx]
    outer_center = np.average(k_data, weights=k_data_weights, axis=0)
    
    return [outer_center]

def ra_random_knn(updates, weights, **kwargs):
    '''randomly sample one point from updates, return weighted avg of k nearest neighbors of it.'''
    k = int(np.floor(kwargs['in_eta']*len(updates)))
    idx = random.randint(0, len(updates)-1)
    print('ra_random_knn, idx of random sampling: ', idx)
    k_data_idx, _ = k_closest(updates, updates[idx], k)

    k_data = [updates[i] for i in k_data_idx]
    k_data_weights = [weights[i] for i in k_data_idx]
    return [np.average(k_data, weights=k_data_weights, axis=0)]

def ra_listDec(updates, weights, **kwargs):
    '''randomly sample n points from updates'''
    num_ra = kwargs['num_ra']
    alpha = kwargs['alpha']
    n = max(1, int(np.floor(alpha*len(updates))))
    ra_ups = []
    candidates = list(np.arange(len(updates), dtype=int))
    for i in range(num_ra):
        idx = np.random.choice(candidates, 1, replace=False)
        # print('ra_listDec, idx of random sampling: ', idx[0])
        if n == 1:
            ra_ups.append(updates[idx[0]])
        else:
            k_data_idx, _ = k_closest(updates, updates[idx[0]], n)
            k_data = [updates[i] for i in k_data_idx]
            k_data_weights = [weights[i] for i in k_data_idx]
            ra_ups.append(np.average(k_data, weights=k_data_weights, axis=0))
            # candidates.remove(k_data_idx)
        if len(candidates) > 1:
            candidates.remove(idx[0])
    return ra_ups

def ra_norm(points, weights, **kwargs):
    points = np.array(points)  # (n, d)
    norms = np.linalg.norm(points, axis=1)  # (n,)
    norm_bound = np.percentile(norms, 90)
    # print('Norm bound: %.4f' % (norm_bound))
    multiplier = np.minimum(norm_bound / norms, 1)  # (n,)
    points = points * multiplier[:, None]
    return [np.average(points, weights=weights, axis=0)]

def k_closest(data, x, k) -> np.array:
    """
    Finds the k closest points to x in data

    Input:
        data (array like): data
        x (np.array): point to find k closest points in data to
        k (int): number of points closest to x to find
    
    Return:
        k_data (np.array): k points in data that are closest to x
        key_dist (float): maximum distance from x to points in k_data
    """
    n = len(data)
    # distance from x to every point
    distances = [np.linalg.norm(x-point) for point in data]

    sorted_distances = sorted(distances)

    # the distance where points with distances lower than this are the k closest
    key_dist = sorted_distances[k-1]

    k_data_idx = np.array([i for i in range(n) if distances[i] <= key_dist])

    return k_data_idx, key_dist


# dictionary of functions whose name starts with "alg__" (i.e. the ones in this file)
aggrs = {name: func for name, func in locals().copy().items() if name.startswith("ra_")}