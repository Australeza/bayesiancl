from typing import Any

import numpy as np
from numpy import signedinteger, ndarray, dtype

from bayesian_clustering.priors import normal_conv


def compute_invconstant(values:np.array) -> signedinteger[Any]:
    """
    Inverse of the normalizing constant.
    Parameters
    ----------
    values: np.array[float]
        prior lambda values

    Returns
    -------
    float
        Product of values from 1d array  + 1
    """
    if values.shape[0] != 1:
        small_l = np.sum(values, axis=0) + 1
    else:
        small_l = values + 1
    return np.prod(small_l)


def omega(l_values:np.array) -> np.ndarray:
    r"""
    Computes omega matrix/ posterior probabilities.

    Parameters
    ----------
    l_values: np.ndarray
        prior lambdas

    Returns
    -------
    np.ndarray
        (k)x(n) values for omega based on the following formula:
         \omega_{1,i} = \frac{\lambda_{1,i}}{\lambda_{1,i}+1}\, , \quad \quad \omega_{2,i} = 1 - \omega_{1,i}\, , \quad{\small i \in [n]}
    """
    denom = np.sum(l_values, axis=0) + 1
    omega_wo_k = np.divide(l_values, denom)
    omega_k = -(np.sum(omega_wo_k, axis=0) - 1)
    omega_final = np.block([[omega_wo_k], [omega_k]])
    return omega_final

#duplicate to margs_mats
def get_pxi(sample_x:list, prior_mus:np.array, func = normal_conv) -> ndarray[
    Any, dtype[Any]]:
    r"""
    Computes probability matrix P_{x,i}

    Parameters
    ----------
    sample_x: list
        observed data
    prior_mus: list
        prior mean values for each cluster
    func: function
        distribution of data convoluted with prior distribution
        default normal_conv: data prior distribution is normal
        alternatives, gamma_conv: data prior distribution is gamma
    Returns
    -------
    np.matrix
        (k)x(n) probabilities for (point i, cluster l) assuming normal prior.
    """
    pxs_d = np.array([[func(x_i, mu) for x_i in sample_x] for idx_mu, mu in enumerate(prior_mus)])
    return pxs_d


def normalizing_const(px:np.array, omegas:np.array) -> tuple:
    """
    Computes normalizing constant.

    Parameters
    ----------
    px: np.array
        (n)x(k) probability matrix
    omegas: np.array
        mixture probabilities

    Returns
    -------
    tuple
        np.array: N-dimensional normalizing constant(useful for post probabilities).
        float: Full normalizing constant (Px).
    """
    pairs = np.multiply(np.array(px), omegas)
    n_sums = np.sum(pairs, axis=0)
    # print(n_sums)
    return n_sums, np.prod(n_sums)


def all_pmarginals(c_const:float, px_mat:np.matrix, prior_matrix:np.matrix, part_space:list) -> tuple:
    """
    Computes marginal_i times lambda_i for all partitions in product form context.

    Parameters
    ----------
    c_const: float
        constant
    px_mat: numpy.matrix
        (n)x(k) probability matrix
    prior_matrix: np.ndarray
        prior lambdas matrix
    part_space: list
        partition space

    Returns
    -------
    tuple
        Values for each partition in the partition space

    """
    lambdas_pspace = []
    px_pspace = []
    px_prod = []
    for partition in part_space:
        part_lambda, part_px, prod_i = prod_lam_marginal(partition,px_mat,prior_matrix)
        lambdas_pspace.append(part_lambda)
        px_pspace.append(part_px)
        px_prod.append(prod_i)
    return np.array(lambdas_pspace) * 1. / c_const, px_pspace, px_prod


def prod_lam_marginal(space_i: list, px_mat: np.matrix, prior_matrix:np.matrix) -> tuple:
    """
    Marginal_I times lambda_I for a single partition in product-form context.

    Parameters
    ----------
    space_i: list
        single partition
    px_mat: np.matrix
        (n)x(k) probability matrix
    prior_matrix: np.ndarray
        prior lambdas

    Returns
    -------
    tuple
        Marginal_I, lambda_I, Marginal_I times lambda_I
    """
    marginal = 1
    lambdas_li = 1
    for cl_idx, cluster in enumerate(space_i):
        for obs in cluster:
            marginal *= px_mat[cl_idx, obs - 1]  # marginal value
            if cl_idx != len(space_i) - 1:
                lambdas_li *= prior_matrix[cl_idx, obs - 1]  # lambda_I
    return lambdas_li, marginal, marginal * lambdas_li
