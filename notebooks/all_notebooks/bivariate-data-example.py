# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:thesis] *
#     language: python
#     name: conda-env-thesis-py
# ---

# ## Product-form - bivariate data
# ---
# This notebook contains the implementation of Bayesian Clustering model for k groups using a prior in product-form. 
#
# **Table of content**
# 1. [Section 1](#section1)
# 2. [Section 2](#section2)
# 3. [Introduction](#section3)<details> <summary><i> Click here for subsection </i></summary><b> [Introductory Example](#intro-example)</b>
# </details>
#
# ---

# +
# Base Python imports
from itertools import combinations
from typing import Any
from more_itertools import locate, random_combination_with_replacement

import operator
import time
import random
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

# Bayesian_clustering package imports
from bayesian_clustering.data_simulation import simulate_2d_data, plot_points, partition_space_nulls
from bayesian_clustering.priors import normal_conv, sample_uniprod, check_sums, normal_conv_2d,  g_multicard, anal_norm_conv_2d
from bayesian_clustering.product_form import compute_invconstant, omega,  get_pxi, normalizing_const, all_pmarginals
from bayesian_clustering.prediction_post import prod_post_probs, partition_undo1d, prod_max_post_prob,partition_to1d_labels, repeat_sampling, prod_random_partition,card_post_probs,card_k_comp_post,card_max_cluster,card_random_partition
from bayesian_clustering.card_based import margs_mat, get_klambdas, mult_polynomial, get_multic, fill_card, binom_op
# %load_ext autoreload
# %autoreload 2
# -

# ## 2-dim clustering
#
# Suppose that the data are two dimentional points generated as follows,
#
# $$\mathbf{X}_i, \sim \mathcal{N}(\mathbb{\theta}_i, \mathbb{\Sigma}), \quad \small \text{ for } i\in [k] $$
#
# where k is the number of clusters, $\Sigma = \mathbb{I}_2 = \begin{bmatrix}
# 1 & 0\\
# 0 & 1
# \end{bmatrix}$ is the identity matrix and
# $$\theta_i = (\theta_i^1, \theta_i^2)|I\quad  \sim \Pi_{I, i} $$ 
# where $\Pi_{l} = \bigotimes_{l\in k}\bigotimes_{i\in I_l} \Pi_{I_l,i}$.
#

# #### Given that $I$ is random and unknown,
#
# $$\theta \sim \sum_{I} \lambda_I \Pi_{I}$$
#
# with $\lambda_I \in [0,1]$ and $\sum_I \lambda_I = 1$.
#
# $\lambda$ is the prior on the partitions space which is considered the exact same as in 1 dimension for unknown I.
#
# The formulas don't change at all.

# ### Simulation of 2-d data points

# - true means and true partition

# +
K = 3 #number of clusters
N = 100 #number of points

true_means = [(3,1), (5,7), (2,4)]

label = list(random_combination_with_replacement(range(K), N))
random.shuffle(label)
partition = list(partition_undo1d(list(label), K))
label[:4], partition[:2][0]
# -

# - data generation

mixture_2d, sample_2d = simulate_2d_data(true_means, partition, N)

sample_2d[:4,:]

# - plot

# - $\lambda$ for individual points

# +
#generate matrix (K-1)xN *uniform* distribution
values_L = sample_uniprod(N,K)

#print resulted matrix
values_L.shape, values_L[:2,:3]
# -

# - constant $C_{\lambda}$

c_lambda_inv = compute_invconstant(values_L)
c_lambda_inv

# - posterior $\omega$'s

omega_li = omega(values_L)
print("shape omega", omega_li.shape, " lambda matrix", values_L.shape)

# - ### significant change matrix $P_{X,I,i}$

prior_mus = np.array([[2, 1.5], [4, 8], [1, 4.6]]) #prior means for theta distribution

# +
#px_inv = get_pxi(sample_2d, prior_mus, normal_conv_2d).T; #px_inv.T
#normal_conv_2d (using integration) is the only difference to 1d approach

px_inv = get_pxi(sample_2d, prior_mus, anal_norm_conv_2d) #analytical formula
px_inv.T.shape
# -

# - constant $P_x$

px_sums, px_constant = normalizing_const(px_inv, omega_li)
px_constant

# ### Parsing partition space
# - if N>12 skip the following method, jump to "Using the postetior"
# - It is not computationally possible to produce the partition space

# +
#partition_space = partition_space_nulls(N,K) computationally costly

# +
#lambdas_space, px_space, prod_space = all_pmarginals(c_lambda_inv, px_inv.T, values_L, partition_space)
#product to-be-maximized
#normalized_pxs = np.array(prod_space)*1./(c_lambda_inv*px_constant)
#normalized_pxs[:3]
# -

# - resulted partition

# +
#i_max = np.argmax(normalized_pxs)
#i_max, partition_space[i_max], normalized_pxs[i_max], partition
# -

# - assumptions check

# +
#check_sums(lambdas_space)
# -

# ### Using posterior

ps = prod_post_probs(omega_li, px_inv, px_sums)
np.sum(ps, axis=0), type(ps)

# - maximum resulted cluster

max_part = prod_max_post_prob(ps)
max_part[:4]

# - random sampling

# +
all_occur = repeat_sampling(200, ps, prod_random_partition); 

# Picking the partition with max sampling occurence
max_occ_k = max(all_occur.items(), key=operator.itemgetter(1))[0]
max_occ_k[:6], type(all_occur)#,  all_occur
# -

print("Random Sampling: estimates_partition", partition_undo1d(max_occ_k, K)," true_partition:", partition)

# - plot with predictions

import seaborn as sns

X = np.array(sample_2d)
true_label = np.array(label) +1
predicted_labels = np.array(max_occ_k)+1

# +
fig, axes = plt.subplots(1,2, figsize = (12,5))

sns.scatterplot(x = X[:,0], y = X[:,1], hue = predicted_labels, palette = "rocket", s=100, edgecolor="black", ax = axes[0])
axes[0].set_title("Predicted clustering")
axes[0].legend(title="cluster")

sns.scatterplot(x = X[:,0], y = X[:,1], hue = true_label, palette = "rocket", s=100, edgecolor="black", ax = axes[1])
axes[1].set_title("True clustering")
axes[1].legend_.remove()

# -
# # Cardinality Based

# - new data generation

# +
K2 = 3 #number of clusters
N2 = 10 #number of points

true_means_2d = [(3,1), (5,7), (2,4)]

label_2d = list(random_combination_with_replacement(range(K2), N2))
random.shuffle(label)
partition_2d = list(partition_undo1d(list(label_2d), K2))
label_2d, partition_2d
# -

mixture_2d, sample_2d = simulate_2d_data(true_means_2d, partition_2d, N2)

sample_2d

# - prior means

prior_means_2d = [[1,0], [3,4], [3,7]]

priors_2d = margs_mat(sample_2d, prior_means_2d, anal_norm_conv_2d)
priors_2d.shape, priors_2d

# - the multivariate polynomial

polyn_2d = mult_polynomial(priors_2d)

coeffs_2d = get_multic(polyn_2d)
coeffs_2d = dict(map(lambda kv: (fill_card(N2, kv[0]), kv[1]),coeffs_2d.items()))
list(coeffs_2d.items())[:3]

prior_vs2 = g_multicard(N2, coeffs_2d.keys(), [0.2, 0.1, 0.7])
list(prior_vs2.items())[:3]


# +
def get_prior_per_m(prior_values:list, all_ms:list)->dict:
    prior_dict = dict.fromkeys(all_ms)
    for idx, m in enumerate(all_ms):
        prior_dict[m] = prior_values[m]
    return prior_dict
       
g_per_m2 = get_prior_per_m(prior_vs2, coeffs_2d)
list(g_per_m2.items())[:3]
# -

# - compute $\lambda_m $

lambdas_m2 = { m: g_per_m2[m]*1./binom_op(m) for m in g_per_m2.keys()}

# - constant $P_X$

px2 = np.inner(np.array(list(coeffs_2d.values())),np.array(list(lambdas_m2.values())))
px2

# - Using posterior

p_k2 = card_post_probs(N2, priors_2d, px2, lambdas_m2, card_k_comp_post)
print(list(map(list, p_k2)), len(p_k2))
p_k2[:4]

# - maximum posterior probability

max_occ_2d = card_max_cluster(p_k2.tolist()); max_occ_2d

# - random sampling

# +
#Sampling from posterior probabilities
all_occur_k2 = repeat_sampling(200, p_k2.tolist(), card_random_partition);

#Partition with maximum occurence
max_rep_2d = max(all_occur_k2.items(), key=operator.itemgetter(1))[0]
# -

max_rep_2d

# Back to lists representation
est_partition = partition_undo1d(max_rep_2d, K2)
est_partition

# - plot

import seaborn as sns

X2 = np.array(sample_2d)
true_labels = np.array(label_2d)
predicted_labels = np.array(max_rep_2d)

# +
fig, axes = plt.subplots(1,2, figsize = (12,5))

sns.scatterplot(x = X2[:,0], y = X2[:,1], hue = predicted_labels, palette = "rocket", s=100, edgecolor="black", ax = axes[0])
axes[0].set_title("Predicted clustering")
axes[0].legend(title="cluster")

sns.scatterplot(x = X2[:,0], y = X2[:,1], hue = true_labels, palette = "rocket", s=100, edgecolor="black", ax = axes[1])
axes[1].set_title("True clustering")
axes[1].legend_.remove()
# -




