# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## k groups cardinality-based prior
#
# ---
# This notebook contains the **partition space** approach for $K\geq 2$.
#
# **Table of content**
# 1. [Simulation of data](#section1)
# 2. [Method in steps](#section2)
# 3. [True partition estimate](#section3) 3. [Introduction](#section3)<details> <summary><i> Click here for subsection </i></summary><b> [Introductory Example](#intro-example)</b>
# </details>
#
# Note that:
# 1. Priors means should be initialized appropriately.
# 2. For big N, it is very computationally costly
#
#
# ---

# ### Imports

# +
# Base Python imports
from itertools import combinations
from typing import Any
from more_itertools import locate
import operator
import time
from collections import Counter

# Other packages/requirements
import scipy.stats as sps
import sympy as sp
import scipy.special
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

# BC package imports 
from bayesian_clustering.data_simulation import gen_partitions, simulate_data, plot_points,  partition_space_nulls
from bayesian_clustering.priors import g_normcard, g_multicard, check_sums, permute_combs
from bayesian_clustering.card_based import margs_mat, get_1d_polynomial, get_coeffs, get_subsets_m, all_cmarginals, get_klambdas, get_constant_px, get_argmax_px, fill_cluster, mult_polynomial, get_multic, fill_card, binom_op
from bayesian_clustering.prediction_post import card_post_probs, repeat_sampling, partition_to1d_labels, card_random_partition, card_max_cluster, partition_undo1d, card_comp_post, card_k_comp_post
from bayesian_clustering.distances import perfect_match_distance
from bayesian_clustering.card_based import unique, fill_coeffs
# %load_ext autoreload
# %autoreload 2

# +
#setting seed for reproducibility 
np.random.seed(160)

K = 2 #number of clusters
N = 7 #number of observations

#true means
true_means = [2, 10]

#prior means
prior_mus = [1, 9]

#generate all possible k-groups w/ permutations
partition_space = partition_space_nulls(N, K)

#randomly select the true partition
true_idx = np.random.choice([i for i in range(len(partition_space))])

true_partition = partition_space[true_idx]
true_partition
# -

#simulate target vector x of N observations
mixture, sampled_x = simulate_data(true_means, true_partition, N)
sampled_x

plot_points(sampled_x, true_means, K)

priors = margs_mat(sampled_x, prior_mus)
priors.shape, priors, type(priors)

polyn_p = mult_polynomial(priors)
polyn_p

coeffs_p = get_multic(polyn_p)
list(coeffs_p.items())[:4], np.argmax(list(coeffs_p.values()))

# 1. unique card - draw prior values
# 2. permute and assign values 1/#permutes
# 3. dictionary of values when permuted same 

filled_coeffs = fill_coeffs(N, coeffs_p)
list(filled_coeffs.items())[:4]

unique_coeffs = unique(filled_coeffs)
list(unique_coeffs.items()), type(unique_coeffs)

print(list(unique_coeffs)[0])

# generate prior values of size len(cardinalities)
prior_vs = g_multicard(N, unique_coeffs, [0.7, 0.3])
#list(prior_vs.items())[:10]
np.sum(list(prior_vs.values()))

partition_space[:10]

lambdas_M = get_klambdas(prior_vs)

marginals_per_partition, lambdas_pp = all_cmarginals(priors, partition_space, lambdas_M)

list(lambdas_M.items())[:4], list(lambdas_I)[-4:]

# ### Constant Px

marginal_value = get_constant_px(list(filled_coeffs.values()), list(lambdas_M.values()))
marginal_value

# ### Partition estimate

max_idx, px_max = get_argmax_px(marginals_per_partition, lambdas_I)
partition_space[max_idx], true_partition

# ### Assumptions check

check_sums(prior_vs.values(), lambdas_I)

plot_points(sampled_x, true_means, K)












