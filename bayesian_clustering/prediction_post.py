from typing import Any

import numpy as np
from more_itertools import locate
from collections import Counter
from collections.abc import Callable

from numpy import matrix

from bayesian_clustering.card_based import get_1d_polynomial, get_coeffs, \
    mult_polynomial, get_multic, fill_card
import operator

### Product form
def prod_post_probs(omega: np.matrix, pxi: np.matrix, constant:np.array)->np.matrix:
    """Posterior probability of point i belonging to cluster l.

    Parameters
    ----------
    omega: np.matrix
        posterior probability matrix
    pxi: np.matrix
        probability matrix
    constant: np.array
        normalizing constant value for ith data point

    Returns
    -------
    np.matrix
        posterior probability matrix w


    """
    return np.multiply(omega, pxi)/constant

def prod_max_post_prob(probs: np.matrix)->list:
    """Maximum posterior probability of point i belonging to cluster l.

    Parameters
    ----------
    probs: np.matrix
        (n)x(k) posterior probability matrix

    Returns
    -------
    list
        finds the indices that maximize probs which correspond to indices of the cluster
        i.e. for 0th point pick [0.4,0.5,0.1] picks index 1-> 2nd cluster

    """
    result = np.argmax(probs, axis = 0).tolist()
   # if probs is 2d:
    #    result = result[0]
    return result
def partition_undo1d(max_occ:list, k:int)->list:
    """From array of cluster indices to list of lists

    Parameters
    ----------
    max_occ: list
        cluster based on maximum probability
    k:int
        number of clusters

    Returns
    -------
    list
        Converts cluster indices to list of data indices
        i.e. [1,1,0,0]->[[3,4][1,2]]

    """

    partition_idxs = list(max_occ)
    partition = []
    for group in range(k):
        element = np.array(list(locate(partition_idxs, lambda x: x == group)))+1
        partition.append(element.tolist())
    return partition

def prod_random_partition(probs:np.matrix):
    """For each data point, sample from its posterior probabilities

    Parameters
    ----------
    probs: np.matrix
        (n)x(k) posterior probability matrix

    Returns
    -------
    list
        Using the posterior as distribution for point j we sample its cluster
        i.e. For jth point say posterior [0.2, 0.3, 0.5]
            Then r.v. is the cluster of jth point and its distribution is its posterior.
            Result: i.e. jth lands on 1st
        This is done for all points given their posterior, resulting a partition.
    """
    partition = []
    n = probs.shape[1]
    k = probs.shape[0]
    for i in range(n):
        pr = np.transpose(probs[:,i]).tolist()#[0]
        if probs.ndim == 2:
              pr = pr[0]
        partition.append(np.random.choice([j for j in range(k)], p = pr))
    return partition
def repeat_sampling(s:int, probs:np.ndarray, func: Callable)-> Counter:
    """Repeat sampling probability matrix S times.

    Parameters
    ----------
    s: int
        arbitrary choice to repeat sampling
    probs: np.matrix
        (n)x(k) posterior probability matrix
    func: Callable
        either prod_random_partition or card_random_partition

    Returns
    -------
    Counter
        All occurring combinations of clusters and their frequencies
    """
    samples = []
    for i in range(s):
        samples.append(tuple(func(probs)))
    return Counter(samples)


#### Card K = 2
def card_comp_post(i: int, px:float, mat_px: np.ndarray, lambdas_card:list)-> np.ndarray:
    """Posterior probabilities of point i.

    Parameters
    ----------
    i: int
        point of interest
    px: float
        normalizing constant
    mat_px: np.ndarray
        prior matrix
    lambdas_card: list
        lambdas per cardinality

    Returns
    -------
    np.ndarray
        Vector of posterior probabilities of point i belonging to cluster l.
    """
    k = mat_px.shape[0]
    values = np.zeros(k)
    for group in range(k):
        comp_1 = mat_px[group,i]/px
        comp_2 = get_1d_polynomial(mat_px, i)
        coeffs = get_coeffs(comp_2)
        if group == k-1:
            value = np.sum(np.multiply(coeffs, lambdas_card[:-1]))
        else:
            value =np.sum(np.multiply(coeffs, lambdas_card[1:]))
        values[group] = value*comp_1
    return values

def card_post_probs(n:int, priors:np.ndarray, p_x:float, lambdas_card:list)-> \
matrix[Any, Any]:
    """Posterior probabilities of all points.

    Parameters
    ----------
    n: int
        number of observations
    priors: np.ndarray
        prior matrix
    p_x: float
        normalizing constant
    lambdas_card:list
        lambdas per

    Returns
    -------
    matrix
        (n)x(k) posterior probabilities of all points and all clusters.
    """
    if priors.ndim==1:
        func = card_comp_post
    else:
        func = card_k_comp_post
    probs = [func(i, p_x, priors, lambdas_card) for i in range(n)]
    return np.matrix(probs)

def card_max_cluster(raw_probs:np.array)->list:
    """Finds the cluster with the highest posterior probability.

    Parameters
    ----------
    raw_probs: np.array
    (n)x(k) posterior probability matrix

    Returns
    -------
    np.array
        Cluster with the highest posterior probability.
    """
    probs = raw_probs
    max_indexes = list(map(np.argmax, raw_probs))
    ins = []
    if probs.ndim == 1:
        ins = [[idx + 1 for idx in range(len(max_indexes)) if max_indexes[idx] == i] for i in
           range(len(probs[0]))]
    if probs.ndim ==2:
        ins = [[idx + 1 for idx in range(len(max_indexes)) if max_indexes[idx] == i] for
               i in
               range(probs.shape[1])]
    return ins

def card_random_partition(probs:list)->list:
    """Samples from the distribution of each point over the clusters.

    Parameters
    ----------
    probs:list
        posterior probability matrix

    Returns
    -------
    list
        Realized indexes for all data points.
        i.e. [1,0,0] 1st point 2nd cluster
    """
    partition = []
    for i in range(len(probs)):
        partition.append(
            np.random.choice([i for i in range(len(probs[0]))], p=probs[i]))
    return partition

def card_k_comp_post(i: int, px:float, mat_px:np.ndarray, lambdas_m:dict)->np.ndarray:
    """Posterior probabilities of point i for each cluster in the multi-group case.

    Parameters
    ----------
    i: int
        point of interest
    px: float
        normalizing constant
    mat_px: np.ndarray
        prior matrix
    lambdas_m: dict
        lambdas per cardinality

    Returns
    -------
    np.ndarray
        Posterior probabilities of ith point. First filling the 1st cluster and then
        mapping the new multinomial coefficients (lm_times_coeff) to the correct values
        of lambdas i.e. For i = 0 (removed) with mult. polynomial variables [1 z_1 z_2],
        the degree of the polynomial is d(pol) = n else if i >1, d(pol) = n-1.

    """
    k = mat_px.shape[0]
    n = mat_px.shape[1]
    values = np.zeros(k)
    poly_minus_i = mult_polynomial(mat_px, i)

    if i == 0:
        coeffs_i = get_multic(poly_minus_i,n)
    else:
        coeffs_minus_i = get_multic(poly_minus_i, n, raw=True)
        coeffs_i = {fill_card(n - 1, k): v for k, v in coeffs_minus_i.items()}

    for group in range(k):
        comp_1 = mat_px[group, i] / px
        value = lm_times_coeff(i, coeffs_i, lambdas_m, group)
        values[group] = value * comp_1
    #print(values)
    return values


def helper_pick_k(k: tuple, group_i:int)-> tuple:
    """Increases group_i-th coordinate of carninality tuple k.

    Parameters
    ----------
    k: tuple
        cardinalities
    group_i: int
        cluster index

    Returns
    -------
    tuple
        New cardinalities tuple where the group_i-th coordinate is increased by 1.
    """
    list_k = list(k)
    list_k[group_i] = list_k[group_i]+1
    return tuple(list_k)


def lm_times_coeff(i: int, coeffs:dict, lambdas_m:list, group: int)->float:
    """

    Parameters
    ----------
    i: int
        point of interest
    coeffs: dict
        coefficients of the mult polynomial
    lambdas_m: list
        lambdas per cardinality
    group: int


    Returns
    -------
        float
            probability of point i belonging to cluster "group".
    """
    if i == 0:
        resulted_dict = {k: coeffs[k] * lambdas_m[k] for k, v in coeffs.items()}
    else:
        resulted_dict = {k: coeffs[k] * lambdas_m[helper_pick_k(k, group)] for k, v in
                         coeffs.items()}
    return sum(list(resulted_dict.values()))

def get_max_occur(all_occurences):
    return max(all_occurences.items(), key=operator.itemgetter(1))

def partition_to1d_labels(n:int, partition:list[list[int]])->np.array:
    """Flattens partition to labels.

    Parameters
    ----------
    n: int
        number of observations
    partition: list of lists of int
        partition of multi-group case

    Returns
    -------
    np.array
        1-d labels corresponding to each partition.
        i.e. [[4,2][1,3]] -> [1,0,1,0]
    """
    labels = np.zeros(n, dtype=int)
    for idx, cluster in enumerate(partition):
        for x in cluster:
            labels[x-1] = idx
    return labels
