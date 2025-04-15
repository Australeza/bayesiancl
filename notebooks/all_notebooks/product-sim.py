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

# ## Simulation product-form
# ---
# This notebook contains the implementation of Bayesian Clustering model for **$K\geq 2$ groups** using a prior on the partition space in product-form for **single-variate** data. It contains, the **Partition Space** Approach and the **Posterior Prediction** approach. 
#
# **Table of content**
# 1. [Initialization](#section1)
# 2. [Partition Space Approach](#section2)
# 3. [Posterior-prediction Approach](#section3)
# </details>
#
# Note: In this example, the choice of the following variables is arbritrarily fixed; the true means, the priors means, the number of points and the number of clusters.
#
# ---

# ## **Imports**

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

# Bayesian_clusterign package imports
from bayesian_clustering.data_simulation import simulate_data, plot_points, partition_space_nulls
from bayesian_clustering.priors import normal_conv, sample_uniprod, check_sums
from bayesian_clustering.product_form import compute_invconstant, omega,  get_pxi, normalizing_const, all_pmarginals
from bayesian_clustering.prediction_post import prod_post_probs, partition_undo1d, prod_max_post_prob, repeat_sampling, prod_random_partition

# %load_ext autoreload
# %autoreload 2
# -

# ## Model assumptions

# +
K = 3 #number of clusters
N = 7 #number of observations

#true means
true_means = [2, 7, 10]

#prior means
prior_mus = [1, 8, 9]
# -

# ### Generating partition space incl. empty clusters

partition_space = partition_space_nulls(N,K)  
partition_space[-3:]

# ### True partition

# +
#randomly pick a partition from the partition space
true_idx = np.random.choice([i for i in range(len(partition_space))])
true_partition = partition_space[true_idx]

true_partition
# -

# ### Simulating data

# +
#simulate target vector x of N observations
mixture, sampled_x = simulate_data(true_means, true_partition, N)

sampled_x
# -

# ### Visualize the points

plot_points(sampled_x, true_means, K)

# ## Generate Matrix $\mathbf{\Lambda}$
# $$\mathbf{\Lambda}= \begin{bmatrix}
# \lambda_{1,1}& \cdots & \lambda_{1,n-1} & \lambda_{1,n} \\
# \lambda_{2,1} & \cdots & \lambda_{2,n-1} & \lambda_{2,n} \\
# \vdots & \ddots & \ddots & \vdots \\
# \lambda_{k,1} & \cdots & \cdots & \lambda_{k,n}
# \end{bmatrix}$$
#
# $\sum_l\lambda_{l,i} =1, \quad l \in [k]$
#
# $\lambda_{lj}$ corresponds to probability that observation $j\in [n]$ belongs to group $l \in [k]$. 

# +
#generate matrix (K-1)xN *uniform* distribution
values_L = sample_uniprod(N,K)

#print resulted matrix
values_L.shape


# -


# ### Normalizing constant $C_{\lambda}$
# *its reverse is computed here*

c_lambda = compute_invconstant(values_L)
c_lambda

# ## Computing $\Omega$

# $$\omega_{l,i}= \frac{\lambda_{l,i}}{\sum_{s=1}^{k-1} \lambda_{s,i}+1}, \quad \omega_{k,i} = 1 -\sum_{s\in [k-1]} \omega_{s,i}\, , \quad\small{l \in [k-1], \; i \in [n]}$$
#
# $$\mathbf{\Omega} = (\omega_{li})= \begin{bmatrix} \omega_{1,1} & \omega_{1,2} & \dots & \omega_{1,n} \\
#  \omega_{2,1} & \omega_{2,2} & \dots & \omega_{2,n} \\
# \vdots & \ddots & \dots & \vdots \\
#  \omega_{k,1} & \omega_{k,2} & \dots & \omega_{k,n}
#  \end{bmatrix} $$

omega_li = omega(values_L)
print(omega_li, "\n lambda matrix\n", values_L)

# ### Computing **$P$**

# $$\mathbf{P}^T = \begin{bmatrix} P_{X,I_1}(x_1) & P_{X,I_1}(x_2) & \dots & P_{X,I_1}(x_n) \\
#  P_{X,I_2}(x_1) & P_{X,I_2}(x_2) & \dots & P_{X,I_2}(x_n) \\
# \vdots & \ddots & \dots & \vdots \\
#  P_{X,I_k}(x_1) & P_{X,I_k}(x_2) & \dots & P_{X,I_k}(x_n) 
#  \end{bmatrix}^{T} $$

px_inv = get_pxi(sampled_x, prior_mus).T
px_inv.T

# ### Computing constant $P_X(X)$

px_sums, px_constant = normalizing_const(px_inv.T, omega_li)
px_constant

# ## **1.**  *Partition Space*- prediction

# ### *Computing $\lambda_I$ and marginals*
# *their product is computed s.t. to be maximized*

lambdas_space, px_space, prod_space = all_pmarginals(c_lambda, px_inv.T, values_L, partition_space)
#product to-be-maximized
normalized_pxs = np.array(prod_space)*1./px_constant
normalized_pxs[:3]

# ### *Resulted partition*

i_max = np.argmax(normalized_pxs)
i_max, partition_space[i_max], normalized_pxs[i_max], true_partition

# ### *Assumptions check*
# $$\sum_{\mathbf{I} \in \mathcal{I_k}} \lambda_{\mathbf{I}} = 1$$

check_sums(lambdas_space)

# ## **2.**  *Posterior* prediction

# $$ p_{l*,i} = \frac{\omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} \quad \text{ for } l* \in [k-1] \quad \text{ and } \quad p_{k,i} = \frac{(1- \sum_{l \in [k-1]} \omega_{s,i})\cdot P_{X,I_{k},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} $$


ps = prod_post_probs(omega_li, px_inv.T, px_sums)
np.sum(ps, axis=0), type(ps), ps[:,1], ps.shape[0], ps.shape[1]

# #### Maximum cluster

max_part = prod_max_post_prob(ps)
max_part

print("Maximum cluster: estimated_partition", partition_undo1d(max_part, K)," true_partition:", true_partition)

# #### Random sampling

# +
all_occur = repeat_sampling(200, ps, prod_random_partition); 

# Picking the partition with max sampling occurence
max_occ_k = max(all_occur.items(), key=operator.itemgetter(1))[0]
max_occ_k, all_occur, type(all_occur)
# -

print("Random Sampling: estimates_partition",partition_undo1d(list(max_occ_k), K)," true_partition:", true_partition)

plot_points(sampled_x, true_means, K)




