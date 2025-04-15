import itertools
from typing import Any
from itertools import combinations

import numpy as np
import scipy
import scipy.stats as sps
from numpy import dtype, ndarray
import sympy as sp

from bayesian_clustering.priors import normal_conv



# Base for all cases K = 2, K > 2
def margs_mat(sample_x: list[float], prior_mus: list, func = normal_conv)-> ndarray[Any, dtype[Any]]:
    r"""Probability matrix of dimensions n x k.

    Parameters
    ----------
    sample_x: list[float]
        observed data
    prior_mus: list[float]
        prior/assumed means of the distribution
    func: function
        it resembles the convolution of dist of data with prior distribution
        it can be default 'normal_conv' or 'gamma_conv' imported from .priors
    Returns
    -------
    matrix : ndarray[Any, Any]
        Probabilities for each point i \in [n] belonging to cluster l \in [k].
    """
    n_cl = len(prior_mus)
    n_x = len(sample_x)
    pxs_m = np.array([[func(sample_x[idx], prior_mus[idx_mu]) for idx in range(n_x)] for idx_mu in range(n_cl)])
    return pxs_m


##### K = 2#####
#polynomial
def get_1d_polynomial(mat_priors, *args) -> list:
    r"""Represents the polynomial computed from probability matrix.

    Parameters
    ----------
    mat_priors: np.ndarray[Any, Any]
        probability matrix of point i belonging to cluster k.
    args:
        value to be excluded, used for posterior prediction

    Returns
    -------
    numpy.poly1d
        Computes the polynomial as \Prod_i (mat_priors[0][i]+mat_priors[1][i]) for each obs except i in args
    """
    p_ab = np.poly1d(1)
    size = mat_priors.shape[1]
    for i in range(size):
        if i in args:
            continue
        a = mat_priors[0]
        b = mat_priors[1]
        p_ab *= np.poly1d([a[i], b[i]])  # a_i z + b_i
    return p_ab


def get_coeffs(polyp: np.poly1d)->list:
    """Extracts all useful coefficients of polynomial of degree k.

    Parameters
    ----------
    polyp: np.poly1d
    polynomial of degree k

    Returns
    -------
    list
     Coefficients of the 1st degree monomial up until (k-1)th degree monomial of polynomial of degree k.

    """
    px_coeffs_arr = polyp.c
    px_coeffs = list(px_coeffs_arr)
    px_coeffs.pop(len(px_coeffs_arr)-1)
    px_coeffs.pop(0)
    px_coeffs.reverse()
    return px_coeffs



def all_cmarginals(priors_mat:np.ndarray, subsets: list) -> list[list]:
    f"""Calculating the marginals for each possible partition.

    Parameters
    ----------
    priors_mat: np.ndarray
        probability matrix of dimensions n x k
    subsets: list
        partitions from the partition space

    Returns
    -------
    list
        Marginals for each possible partition by calling cluster_marginal.

    """
    px_all = []
    for partition in subsets:
        px_all.append(cluster_cmarginal(priors_mat, partition))
    return px_all


def cluster_cmarginal(priors_mat:np.ndarray, partition: list) -> list:
    """Computed the marginal for specified partition.

    Parameters
    ----------
    priors_mat: np.ndarray
        probability matrix of dimensions n x k
    partition:  list
        single partition

    Returns
    -------
    float:
        Marginal value of a specified partition.
    """
    marginal = 1
    #if priors_mat.shape[0] == 2:
        #partition = fill_cluster(priors_mat.shape[1], partition)
      #  print(partition)
    for cl_idx, cluster in enumerate(partition):
        for obs in cluster:
            marginal *= priors_mat[cl_idx, obs - 1]

    return marginal

# Generate subsets of cardinality m
def get_subsets_m(n:int, m: int) -> np.array:
    """Finds all possible subsets  of size m of n observations for the first/on- cluster.

    Parameters
    ----------
    n: int
        number of observations
    m: int
        cardinality of cluster 1

    Returns
    -------
    np.array
        All possible subsets of size m of n observations.

    """
    obs_ind = [i for i in range(1,n+1)]
    all_subs_m = np.array(list(combinations(obs_ind, m))).tolist()
    return all_subs_m

# Assign lambdas to cardinalities, and to partitions
def get_lambdas(g_list:list, all_subs:list, n:int)-> tuple[list[Any], list[Any]]:
    """Calculating lambdas for each subset and values per cardinality.

    Parameters
    ----------
    g_list: list
        prior values for each cardinality
    all_subs: list
        partition space
    n: int
        number of observations

    Returns
    -------
    tuple[list[Any], list[Any]]
     Lambdas for each subset, Lambdas for each cardinality (g_m/#num_of_subsets).

    """
    lambdas_m = [scipy.special.binom(n,i+1)**(-1)*g_list[i] for i in range(len(g_list))]
    print(len(lambdas_m))
    lambdas_I = [lambdas_m[len(subset)-1] for subset in all_subs]
    return lambdas_I, lambdas_m

def get_klambdas(g_list: dict, all_subs:list)-> tuple[list[Any], dict]:
    """Getting lambdas for each subset for k-groups.


    Parameters
    ----------
    g_list: dict
        prior values for each cardinality
    all_subs: list
        partition space
    n: int
        number of observations
    Returns
    -------
    tuple[list[Any], dict]
     Lambdas for each subset, Lambdas for each cardinality (g_m/#num_of_subsets).
    """""
    lambdas_m = {cardinalities: g_list[cardinalities] * 1. / binom_op(cardinalities) for cardinalities in g_list.keys()}
    lambdas_i = [lambdas_m[get_m(part_i)] for part_i in all_subs]
    return lambdas_i, lambdas_m

def get_m(part_i:list)-> tuple:
    """From partition to cardinalities

    Parameters
    ----------
    part_i:list
        single partition

    Returns
    -------
    tuple
        Cardinalities of partition
        i.e. [[1][2 3]] -> (1,2)s

    """
    lens = [len(i) for i in part_i]
    return tuple(lens)
# Compute constant Px
def get_constant_px(coeffs:list, lambdas:list)->float:
    """Normalizing constant Px.

    Parameters
    ----------
    coeffs: list
        coefficients of the polynomial
    lambdas: list
        lambdas per cardinality (must be computed incorrectly?)

    Returns
    -------
    float
        Constant value of the marginal for the whole partition space.
    """

    p_x = np.sum(np.multiply(coeffs, lambdas))
    return p_x

def get_argmax_px(px_all:np.ndarray, lambdas_i:np.ndarray)-> tuple:
    """Finds maximum value of all marginals for each partition.

    Parameters
    ----------
    px_all:np.ndarray
        marginal values for each possible partition
    lambdas_i: np.ndarray
        lambdas per partition in the partition space


    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Index of the maximum weighted by lambdas_i value, the maximum value

    """
    prod_vec = np.multiply(px_all, lambdas_i)
    idx = np.argmax(prod_vec)
    return idx, prod_vec[idx]

def fill_cluster(n: int, cluster:list)->list:
    """Fills in the second cluster of a partition.

    Parameters
    ----------
    n: int
        number of observations
    cluster: list
        single first cluster

    Returns
    -------
    list
        Whole partition by filling in the second/remaining cluster.
        i.e. [[1,4]] -> [[1,4],[2,3]]
    """
    indexes = [i+1 for i in range(n)]
    rest = [idx for idx in indexes if idx not in cluster]
    return [cluster, rest]

### K>2

def mult_polynomial(priors_mat:np.array, element: int = None) -> sp.polys.polytools.Poly:
    r"""Multinomial created from prior conv matrix.

    Parameters
    ----------
    priors_mat: np.array
        (k)x(n) prior matrix
    element: int = None
        single element to be removed for posterior prediction purposes
    Returns
    -------
    sp.polys.polytools.Poly
        Creates multinomial from the expression np.prod( [1, z_1, ..., z_{k-1}]\dot priors).

    """
    k = priors_mat.shape[0] #[1,2,3]
    var = np.array([1] + list(sp.symbols(f'z1:{k}')))
    prod_all = var@priors_mat
    if element is not None:
        prod_all = np.delete(prod_all, element)
    prod_poly = np.prod(prod_all)
    return prod_poly.as_poly()

def fill_card(n:int, card:tuple)->tuple:
    """Fills in the first cardinality of a partition.

    Parameters
    ----------
    n: int
        number of observations
    card: tuple
        cardinalities for (2,..,k) clusters

    Returns
    -------
    tuple
        Whole partition by filling in the first/on- cluster.
        i.e. n = 4, k = 3: (1,2) ->(1,1,2)
    """
    return (n-sum(card),)+card


def get_multic(poly_n:sp.polys.polytools.Poly) -> dict:
    """Extracts the coefficients of the multivariate polynomial.

    Parameters
    ----------
    poly_n: sp.polys.polytools.Poly
        multivariate polynomial

    Returns
    -------
    dict
        Keys the degrees of each variable and value the coefficient of the respective monomial.
        i.e. n= 6, pol = 0.2*z_1^2*z_2^3 then d = {(2,3): 0.2}
    """
    coeffs = {}
    for monom, coeff in zip(poly_n.monoms(), poly_n.coeffs()):
        coeffs[monom] = coeff
    return coeffs

def binom_op(m_tup:tuple):
    """Factorial for multiple groups.

    Parameters
    ----------
    m_tup: tuple
        cardinalities

    Returns
    -------
    float
        Factorial {n}{m_1,..,m_k}.
    """
    sum_n = sum(m_tup)
    binom_result = 1
    for m_i in m_tup:
        binom_result *= scipy.special.binom(sum_n,int(m_i))
        sum_n -= int(m_i)
    return binom_result


def sorted_px(px_all,lambdas_i):
    """Sorts product vector.

    Parameters
    ----------
    px_all: list
        marginals for all clusterings
    lambdas_i: list
        Weights for all clusterings

    Returns
    -------
    list
        Sorted list of px*lambda values.
    """
    # TODO this is incomplete
    prod_vec = np.multiply(px_all, lambdas_i)
    s_prod = sorted(enumerate(prod_vec), key=lambda x: x[1])
    return s_prod

def fill_coeffs(n:int, coeffs_p:dict)->dict:
    """Fills in the cardinalities in the dictionary coeffs_p

    Parameters
    ----------
    n: int
    number of observations
    coeffs_p: dict
    tuple missing first element, coefficients of respective monomials

    Returns
    -------
    dict
    Keys tuples of cardinalities filled in and values the coefficient of the respective
    monomial.
    i.e. n= 2, coeffs_p={(0,1):0.2, (2,0):0.4, etc.}
    output: {(1,0,1):0.2,(0,2,0):0.4 etc.}

    """
    return {fill_card(n, k): v for k,v in coeffs_p.items()}

def unique(cards:dict)->dict:
    """Permuted cardinalities represented by one

    Parameters
    ----------
    cards: dict
    keys cardinalities, values

    Returns
    -------
    dict
    keys are the representative cardinality of the permuted ones
    i.e. cards = {(0,1,0): 0.2, (1,0,0):0.4, (0,0,1):0.2 etc.}
    output: {(1,0,0):0}

    """
    u = {}
    for idx, card in enumerate(cards):
        l = list(card)
        l.sort()
        u[tuple(l)] = 0
    return u
