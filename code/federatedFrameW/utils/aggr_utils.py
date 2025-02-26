import numpy as np
import random
import torch
import math
import copy


def ra_FedAvg(updates, weights, **kwargs):
    # np.array/median() accepts array_like input, If it is not an array, a conversion is attempted.
    return np.average(updates, weights=weights, axis=0) # np.average：avg = sum(a * weights) / sum(weights)

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
    
    return inner_center

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
    
    return outer_center

def ra_rknn(updates, moms, weights, **kwargs):
    '''randomly sample one point from updates, return weighted avg of k nearest neighbors of it.'''
    assert (len(updates) == len(moms))
    updates = np.asarray(updates)
    moms = np.asarray(moms)
    k = kwargs['num_hls']
    means, ra_moms = [], []
    full_ups, full_moms = copy.deepcopy(updates),copy.deepcopy(moms)

    while len(updates) > 0:
        if len(updates) <= k:
            k_data_idx = [i for i in range(len(updates))]
        else:
            idx = np.random.randint(len(updates))
            k_data_idx, _ = k_closest(updates, updates[idx], k)
        # print(f'len updates/moms/weights: {len(updates),len(moms),len(weights)} len k_data_idx: {len(k_data_idx)}')
        
        k_data = [updates[i] for i in k_data_idx]
        k_moms = [moms[i] for i in k_data_idx]
        k_data_weights = [weights[i] for i in k_data_idx]
        means.append(np.average(k_data, weights=k_data_weights,axis=0))
        ra_moms.append(np.average(k_moms, weights=k_data_weights,axis=0))

        updates = [t for i,t in enumerate(updates) if i not in k_data_idx]
        moms = [t for i,t in enumerate(moms) if i not in k_data_idx]
        weights = [t for i,t in enumerate(weights) if i not in k_data_idx]
    
    idx = np.random.randint(len(means))
    return means[idx],ra_moms[idx]

def ra_meb(updates, moms, weights, **kwargs):
    '''Search for the minimum ball of k points repeatedly until no remaining points, 
    and return a center of ball chosen randomly.'''
    assert (len(updates) == len(moms) and len(updates) == len(weights))
    updates = np.asarray(updates)
    moms = np.asarray(moms)
    k = kwargs['num_hls']
    means, ra_moms = [], []
    full_ups, full_moms = copy.deepcopy(updates),copy.deepcopy(moms)

    while len(updates) > 0:
        # find the MEB
        radii = []
        if len(updates) <= k:
            k_data_idx = [i for i in range(len(updates))]
        else:
            for i in range(len(updates)):
                _, radius = k_closest(updates, updates[i], k)
                radii.append(radius)
            min_r_index = np.argmin(radii)
            k_data_idx,_ = k_closest(updates, updates[min_r_index], k)
        # print(f'2len(radii): {len(radii)} len updates/moms/weights: {len(updates),len(moms),len(weights)} len k_data_idx: {len(k_data_idx)}')
        
        # compute the center
        k_data = [updates[i] for i in k_data_idx]
        k_moms = [moms[i] for i in k_data_idx]
        k_data_weights = [weights[i] for i in k_data_idx]
        means.append(np.average(k_data, weights=k_data_weights,axis=0))
        ra_moms.append(np.average(k_moms, weights=k_data_weights,axis=0))

        # update the point set
        updates = [t for i,t in enumerate(updates) if i not in k_data_idx]
        moms = [t for i,t in enumerate(moms) if i not in k_data_idx]
        weights = [t for i,t in enumerate(weights) if i not in k_data_idx]
    
    idx = np.random.randint(len(means))
    return means[idx],ra_moms[idx]

def tensor_mean(data, weights):
    '''return the weighted mean of data, which is a list of tensors'''
    stacked_tensors = torch.stack(data, dim=0)  # 在新的维度0上堆叠tensor  
    weights_expanded = weights.view(-1, 1).expand_as(stacked_tensors)  # 形状变为[3, 10]  
    weighted_sum = stacked_tensors * weights_expanded  
    weighted_average = weighted_sum.sum(dim=0) / weights.sum()
    return weighted_average

def ra_ListDec(updates, moms, weights, **kwargs):
    '''randomly sampling one point from updates''' 
    if len(updates)==1:
        idx = 0
    else:
        print('ra_ListDec multiple candidate updates!')
        candidates = list(np.arange(len(updates), dtype=int))
        idx = np.random.choice(candidates, 1, replace=False)[0]

    return updates[idx],moms[idx]

def ra_norm(points, weights, **kwargs):
    '''norm bounded mean'''
    points = np.asarray(points)  # (n, d)
    norms = np.linalg.norm(points, axis=1)  # (n,)
    if kwargs['norm'] == 'fix':
        norm_bound = 0.215771  # need to re-compute, referring paper "Can You Really Backdoor Federated Learning?"
    elif kwargs['norm'] == 'adap': # refering paper "Boosting Robustness by Clipping Gradients in Distributed Learning"
        k = int(2 * kwargs['out_eta'] * kwargs['num_hls'])
        norm_bound = sorted(norms)[k]
        # print(f'sorted norms: {sorted(norms)[:(k+1)]}\nNorm bound: %.4f' % (norm_bound))

    multiplier = np.minimum(norm_bound / norms, 1)  # (n,)
    points = points * multiplier[:, None]
    return np.average(points, weights=weights, axis=0)

def ra_CWTM(points,  weights, **kwargs):
    '''coordinate-wised trimmed mean'''
    if kwargs['out_eta'] == 0: # no trimming necessary, return simple mean
        return np.average(points, weights=weights, axis=0)
    points = np.asarray(points)  # (n, d)
    weights = np.asarray(weights)
    aggregated_update = np.zeros_like(points[0])
    # discard at least 1 but do not discard too many
    num_points_to_discard = min(len(points) // 2, math.ceil(len(points) * kwargs['out_eta']))  # //: Divide - returns the integer part of the quotient (rounds down),9//2=4
    for i in range(aggregated_update.shape[0]): # coordinate wise
        values = np.asarray([p[i] for p in points])
        # print(f'ra_trimMean, values.shape: {values.shape}') # 35
        idxs = np.argsort(values)[num_points_to_discard: -num_points_to_discard]
        aggregated_update[i] = np.average(values[idxs], weights=weights[idxs], axis=0)
    return aggregated_update

def ra_CWM(points, weights, **kwargs):
    '''coordinate-wised median'''
    # print(f'points.shape: {len(points)} * {points[0].shape}')
    # u0, u1 = [u[0] for u in points], [u[1] for u in points]
    # res = np.median(points, axis=0)
    # print(f'sorted u0[17]: {sorted(u0)[17]} res[0]: {res[0]}\nsorted u1[17]: {sorted(u1)[17]} res[1]: {res[1]}')
    return np.median(np.asarray(points), axis=0)
    # # `weighted_median` computes median along the last axis, so we need to transpose points
    # return weighted_median(np.array(points).T, np.array(weights)).T

def ra_GM(points, weights, **kwargs):
    """Computes geometric median of points with weights using Weiszfeld's Algorithm
    Code from https://github.com/krishnap25/tRFA
    """
    weights = np.asarray(weights, dtype=points[0].dtype) # / sum(weights)
    median = np.average(points, weights=weights, axis=0)
    # print(f'GM len p[0]: {len(points[0])}, len median: {len(median)}') # GM len p[0]: 115646, len median: 115646
    obj_val = geometric_median_objective(median, points, weights)

    for i in range(kwargs['GM_iter']):
        prev_obj_val = obj_val
        weights = np.asarray([weight / max(1e-5, np.linalg.norm(median - p)) for weight, p in zip(weights, points)], dtype=weights.dtype)
        # weights = weights / weights.sum()
        median = np.average(points, weights=weights, axis=0) 
        obj_val = geometric_median_objective(median, points, weights)
        if abs(prev_obj_val - obj_val) < 1e-6 * obj_val:
            break

    return median

def OLDk_closest(data, x, k) -> np.array:
    """
    Finds the k or more closest points to x in data, taking the k-th distance as threshold.

    Input:
        data (array like): data
        x (np.array): point to find k closest points in data to
        k (int): number of points closest to x to find
    
    Return:
        k_data (np.array): k points in data that are closest to x
        key_dist (float): maximum distance from x to points in k_data
    """
    print('OLDk_closest!')
    n = len(data)
    # distance from x to every point
    Y = x - data
    # print(f'x/y.dtype: {x.dtype, Y[0].dtype}') # (dtype('float64'), dtype('float64'))
    # print(f'Y/x.shape: {len(Y), Y[0].shape, x.shape}') # femnistCNN: (35, (115646,), (115646,))
    distances = [np.inf if (np.any(np.isnan(y))) else np.linalg.norm(y) for y in Y]
    sorted_distances = sorted(distances)
    # the distance where points with distances lower than this are the k closest
    if k >= n: 
        key_dist = sorted_distances[-1] 
        k_data_idx = [i for i in range(n)]
    else:
        key_dist = sorted_distances[k-1] 
        k_data_idx = np.array([i for i in range(n) if distances[i] <= key_dist])
    # print(f'1key_dist: %.4f len(k_data_idx): {len(k_data_idx)}' % (key_dist)) # len(k_data_idx): 0 key_dist: nan

    return k_data_idx, key_dist

def k_closest(data, x, k):
    """
    Finds the k closest points to x in data, breaking ties arbitrarily.

    Input:
        data (array like): data
        x (np.array): point to find k closest points in data to
        k (int): number of points closest to x to find
    
    Return:
        k_data_idx (np.array): indices of k points in data that are closest to x
        key_dist (float): maximum distance from x to points in k_data
    """
    n = len(data)
    # distance from x to every point
    Y = x - data
    distances = [np.inf if (np.any(np.isnan(y))) else np.linalg.norm(y) for y in Y]
    idx_distances = [(i,d) for i,d in enumerate(distances)]
    sorted_distances = sorted(idx_distances, key=lambda item: item[1])
    # the distance where points with distances lower than this are the k closest
    if k >= n: 
        key_dist = sorted_distances[-1][1] 
        k_data_idx = [i for i in range(n)]
    else:
        key_dist = sorted_distances[k-1][1] 
        k_data_idx = [x[0] for x in sorted_distances[:k]]

    return k_data_idx, key_dist

def geometric_median_objective(median, points, weights):
        """Compute geometric median objective."""
        return sum([alpha * np.linalg.norm(median - p) for alpha, p in zip(weights, points)])

# dictionary of functions whose name starts with "alg__" (i.e. the ones in this file)
aggrs = {name: func for name, func in locals().copy().items() if name.startswith("ra_")}