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

# # Clustering in K groups - simulation
#
# ---
# This notebook contains an implementation of bayesian hierarchical model for clustering data in two groups using a cardinality based prior on the partition space by producing the whole partition space; this is a *brute-force* approach which will be the *baseline* approach.
#
# **Table of content**
#
# 1. [2 groups](#Section1)
#     - [Data simulation](#subsection)
#
#     1.2[Method in steps](#section2)
#    
#     1.3 [Partition estimate](#section3)
#
# **Issues to be addressed**
#
# 1. If initialized with 0 it suffers
# 2. If prior values are in different order it suffers
# 3. For big sample size it misclusters one observation *which eventually is not a mistake as one observation is overlapping*
#
# ---

# #### **Imports**

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

# BC package imports TODO: do not use * imports, but be specific 
from bayesian_clustering.data_simulation import gen_partitions, simulate_data, plot_points,  partition_space_nulls, initialize_true
from bayesian_clustering.priors import g_normcard, g_multicard, check_sums
from bayesian_clustering.card_based import margs_mat, get_1d_polynomial, get_coeffs, get_subsets_m, all_cmarginals, get_lambdas, get_constant_px, get_argmax_px, fill_cluster, mult_polynomial, get_multic, fill_card, binom_op
from bayesian_clustering.prediction_post import card_post_probs, partition_to1d_labels, repeat_sampling, card_random_partition, card_max_cluster, partition_undo1d, card_comp_post, card_k_comp_post
from bayesian_clustering.distances import perfect_match_distance

# %load_ext autoreload
# %autoreload 2
# -

# ## K = 2 groups

# ### **Data simulation**
# ---

# +
N = 12 #number of observations
K = 2 #number of clusters

#true means
true_means = [2, 5]
# -

# #### True partition

# +
#setting seed for reproducibility 
np.random.seed(103)
true_labels, true_partition = initialize_true(N, K)

'Cluster to be predicted', true_partition, 'True means', true_means
# -

# #### Generate data

#simulate target vector x of N observations
mixture_x, sampled_x = simulate_data(true_means, true_partition, N)
sampled_x

plot_points(sampled_x, true_means, K)

# ### **Method in steps**
# ---

#prior means
prior_mus = [2, 7]

priors = margs_mat(sampled_x, prior_mus)
priors.shape, priors

poly_n = get_1d_polynomial(priors); print(poly_n)

px_coeffs = get_coeffs(poly_n)
px_coeffs, np.argmax(px_coeffs), poly_n[13], len(px_coeffs)

g_m =  g_normcard(N,K) 

lambdas_card = [scipy.special.binom(N,i+1)**(-1)*g_m[i] for i in range(len(g_m))]

p_x = get_constant_px(px_coeffs, lambdas_card)
p_x

# ## Predictions for 2 groups

# $$p_{1,i}  =\frac{P_{X,1,i}(x_i)}{P_X}\sum_{m =1}^{n} \, g(m) A'_{m-1,-i}(P_{X,1}(x),P_{X,2}(x))
# ,  \quad \quad p_{2,i} = \frac{P_{X,2,i}(x_i)}{P_X} \sum_{m =0}^{n-1} \, g(m) A'_{m,-i}(P_{X,1}(x),P_{X,2}(x))
# $$

# **Note to self** basically when it belongs to 1 then g(m) is the same (you again dont consider 0) but if it belongs to 2 then it is possible that cluster one is not going to have all obs in it so [:-1].

# #### Posterior cluster probabilities

p_c = card_post_probs(N, priors, p_x, lambdas_card, card_comp_post); print(p_c)

# #### - **Maximum cluster** for each data point

card_max_cluster(p_c)

# #### - **Sampling** from given probabilities

all_occurances = repeat_sampling(100, np.array(p_c.tolist()),  card_random_partition)
max_occ = max(all_occurances.items(), key=operator.itemgetter(1))[0]
partition_undo1d(max_occ, K), true_partition


plot_points(sampled_x, true_means, K)

# ## **K groups**

# ### **True partition and means**

# +
#setting seed for reproducibility 
np.random.seed(160)

K_k = 3 #number of clusters
N_k = 10 #number of observations

#true means
true_means_k = [2, 10, 20]

#prior means
prior_mus_k = [1, 12, 14]

#generate all possible k-groups w/ permutations
#partition_space_k = partition_space_nulls(N_k,K_k)

#randomly select the true partition
#true_idx_k = np.random.choice([i for i in range(len(partition_space_k))])

true_labels_k, true_partition_k = initialize_true(N_k, K_k)
true_partition_k
# -

# ### **Generate data**

#simulate target vector x of N observations
mixturek, sampled_xk = simulate_data(true_means_k, true_partition_k, N_k)
sampled_xk

plot_points(sampled_xk, true_means_k, K_k)

priors_k = margs_mat(sampled_xk, prior_mus_k)
priors_k.shape, priors_k

# ### The multivariate polynomial

polyn_p = mult_polynomial(priors_k)

coeffs_p = get_multic(polyn_p)
coeffs_p = dict(map(lambda kv: (fill_card(N_k, kv[0]), kv[1]),coeffs_p.items()))
list(coeffs_p.items())[:3]

# ### $g_m$

prior_vs = g_multicard(N_k, coeffs_p.keys(), [0.2, 0.1, 0.7])
list(prior_vs.items())[:3]


# +
def get_prior_per_m(prior_values:list, all_ms:list)->dict:
    prior_dict = dict.fromkeys(all_ms)
    for idx, m in enumerate(all_ms):
        prior_dict[m] = prior_values[m]
    return prior_dict
       
g_per_m = get_prior_per_m(prior_vs, coeffs_p)
list(g_per_m.items())[:3]
# -

# ###  Computing $\lambda_m $

lambdas_m = { m: g_per_m[m]*1./binom_op(m) for m in g_per_m.keys()}

# ### Constant $P_X$

px = np.inner(np.array(list(coeffs_p.values())),np.array(list(lambdas_m.values())))
px

# ## Prediction for k-groups

# $$p_{l*,i} =  \frac{P_{X,I_{l*},i}(x_i)}{P_X}\sum_{m\in \mathcal{M}}  g(\mathbf{m})  A_{\mathbf{m}_{l*},-i}(P_{X,I_1}(x), P_{X,I_2}(x), \dots, P_{X,I_k}(x)), \quad \quad \text{ for } l* \in [k].$$

# ### Distributions of each point over all clusters
# *I could plot those*

p_k = card_post_probs(N_k, priors_k, px, lambdas_m, card_k_comp_post)
p_k[:,1]

# ### Maximum clustering partition

card_max_cluster(p_k)

p_k

# ### Random sampling prediction

# +
#Sampling from posterior probabilities
all_occur_k = repeat_sampling(200, p_k.tolist(), card_random_partition);

#Partition with maximum occurence
max_occ_k = max(all_occur_k.items(), key=operator.itemgetter(1))[0]
max_occ_k
# -

# Back to lists representation
est_partition = partition_undo1d(max_occ_k, K_k)
est_partition

true_partition_k

plot_points(sampled_xk, true_means_k, K_k)










