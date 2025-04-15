from typing import Any, Iterable

import numpy as np
import more_itertools as mit
import random
import itertools
import matplotlib.pyplot as plt
from numpy import ndarray, dtype, floating
from more_itertools import random_combination_with_replacement
from numpy._typing import _64Bit

from bayesian_clustering.prediction_post import partition_undo1d

#Initialization of Data
def initialize_true(n:int, k:int)->[list,list]:
    """
    This function initializes the true labels and the true partition, randomly.

    Parameters
    ----------
    n: int
        number of points
    k: int
        number of clusters

    Returns
    -------
    [list, list]
        True labels and true partition.

    """
    true_labels = list(random_combination_with_replacement(range(k), n))
    random.shuffle(true_labels)
    true_partition = list(partition_undo1d(list(true_labels), k))
    return true_labels, true_partition

### Partition space
def gen_partitions(n:int, k:int, wp:int =0)->list:
    """
    Partition space generator.

    Parameters
    ----------
    n: int
        number of observations
    k: int
        number of clusters
    wp: int = 0
        permutations indicator

    Returns
    -------
    list
        All possible partitions either permuted wp ==1 or not wp ==0.
    """
    data = [i for i in range(1, n+1)]
    all_sets = list(mit.set_partitions(data,k)) #not order preserving [1][2][3 4] = [3 4][1][2]
    if wp == 1:
        all_sets = permute(all_sets)
    return all_sets

#duplicate in card_based probably
def permute(given_p:list)-> list:
    """
    Adds permutations of each partition to the partition space.

    Parameters
    ----------
    given_p:list
        partition space to be permuted

    Returns
    -------
    list
        Updates partition space by including permutations.
    """
    res_part = []
    for u_part in given_p:
        perm = itertools.permutations(u_part)
        perm = map(list, perm)
        res_part = res_part + list(perm)
    return res_part

# Sample 1-dimensional data points

def sample_norm_data(mus: list[float], sizes:list[int], sigma: float = 1)->list:
    """
    Samples from the normal distribution with mean in mus of size in sizes
    Parameters
    ----------
    mus: list[float]
        true means
    sizes: list[int]
        number of observations sampled from each Normal w the respective true mean
    sigma: float = 1
        standard deviation of the Normal distribution

    Returns
    -------
    list
        Lists of normal data sampled from the respective mean
    """
    samples = []
    for k in range(len(mus)):
        samples.append(list(np.random.normal(mus[k], sigma, sizes[k])))
    return samples

def simulate_data(mus:list, true_part: list, n:int, sigma:float = 1)-> \
tuple[Iterable, ndarray[Any, dtype[floating[_64Bit]]]]:
    """
    Simulate data by combining indexes of true partition and sampled_norm_data

    Parameters
    ----------
    mus: list[float]
        true means
    true_part: list
        true partition
    n: int
        number of observations
    sigma: float = 1
        standard deviation of the Normal distribution

    Returns
    -------
    tuple
        list
            Mixture of data as sampled from sample_norm_data.
        np.array
            Simulated data array.
    """
    cluster_len = [ len(cluster) for cluster in true_part]
    mixture = sample_norm_data(mus, cluster_len, sigma)
    x = np.zeros(n)
    for idx, cluster in enumerate(true_part):
        for idx_obs, obs in enumerate(cluster):
            x[obs - 1] = mixture[idx][idx_obs]
    return mixture, x

#Sample 2-dimensional data points
def sample_2d_norm_data(mus:list[float], sizes:list[int], sigma: np.array = np.array([[1,0],[0,1]]))->list:
    """
    Creates 2D samples from the normal distribution with mean in mus and covariance matrix the identity of size equal to the sizes of each cluster.

    Parameters
    ----------
    mus: list[list[float]]
    multi-dimensional (2d) prior means

    sizes: list[int]
    sizes of each cluster

    sigma: np.array
    identity matrix - 2d

    Returns
    -------
    list
    Lists of 2d normal data sampled from the respective mean and covariance matrix.
    i.e. if len(cluster_1) = 1 and len(cluster_2) = 3 and means = [(1, 3), (5, 2)]
        the output is 1 sample from the normal with mean(1,3) and cov I
        and 3 samples from the normal with mean(5,2) and cov I
    """

    samples = []
    for k in range(len(mus)):
        draw = np.random.multivariate_normal(mus[k], sigma, size=sizes[k])
        samples.append(list(draw))
    return samples


def simulate_2d_data(mus:list[float], true_part:list, n:int, sigma:np.array = np.array([[1,0],[0,1]]))-> \
tuple[Iterable, ndarray[Any, dtype[floating[_64Bit]]]]:
    """
        Simulate data by combining indexes of true partition and sampled_norm_data

        Parameters
        ----------
        mus: list[float]
            true means
        true_part: list
            true partition
        n: int
            number of observations
        sigma: float = 1
            standard deviation of the Normal distribution

        Returns
        -------
        tuple
            list
                Mixture of data as sampled from sample_norm_data.
            np.array
                Simulated data array.
        """
    cluster_len = [len(cluster) for cluster in true_part]
    mixture_2d = sample_2d_norm_data(mus, cluster_len, sigma)
    x = np.zeros((n,2))
    for idx, cluster in enumerate(true_part):
        for idx_obs, obs in enumerate(cluster):
            x[obs - 1] = mixture_2d[idx][idx_obs]
    return mixture_2d, x

### Plot points

def plot_points(sample, true_means, k):
    r"""
    Plot of the raw data points.
    Parameters
    ----------
    sample: np.array
        observed sample/ data
    true_means: list
        true means
    k: list
        number of clusters

    Returns
    -------
    None

    """
    plt.figure(figsize=(8,2))
    plt.plot([i-2 for i in range(40)], [0 for i in range(40)], 'black')
    plt.plot(sample, [0 for i in range(len(sample))], marker = 'x', ms = 10, color ='black', mec = 'r')
    plt.ylim(-0.3, 0.3)
    plt.xlim(true_means[0]-2.5, true_means[k-1] + 2.5)
    for i, x in enumerate(sample):
        plt.text(x,0-0.1,i+1, ha="center", va="center")
    plt.show()
    return


### Product-form K>2
### K = 2 and K>2
def add_empty(y: list, empties: list) -> list:
    """
    Extending y with empty elements of empties.

    Parameters
    ----------
    y: list
        partition
    empties: list
        list of empty lists

    Returns
    -------
    list
        Partition extended with empty lists.
    """
    y.extend(empties)
    return y


def empty_subspaces(n: int, k: int, num_empties: int) -> list:
    """
    Partition space considering fixed number of empty clusters.

    Parameters
    ----------
    n:int
        number of observations
    k: int
        number of clusters
    num_empties: int
        number of empty clusters

    Returns
    -------
    list
        Partition space considering num_empties number of empty clusters.

    """
    unique_part = gen_partitions(n, k - num_empties)
    empty_elements = [[] for _ in range(num_empties)]
    empty_parts = list(map(lambda x: add_empty(x, empty_elements), unique_part))
    permutes = permute(empty_parts)
    subspace = [part for idx, part in enumerate(permutes) if
                permutes.index(part) == idx]
    return subspace


def partition_space_nulls(n:int, k:int)->list:
    """
    Partition space considering empty clusters.

    Parameters
    ----------
    n: int
        number of observations
    k: int
        number of clusters

    Returns
    -------
    list
        All possible partitions considering empty clusters.
    """
    partition_space = []
    for i in range(k):
        empties = empty_subspaces(n, k, i)
        partition_space.extend(empties)
    return partition_space

