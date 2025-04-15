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

# ## Priors and distances
# ---
# This notebook will address the options of priors considered in this thesis both for cardinality and product form case as well as for clustering into k=2 and k>2 groups.
#
# Different distances are used between two different labelings to measure the similarity of predictions from truth.
#
# **Table of content**
# 1. [Normal](#section1)
# 2. [Gamma](#section2)
# 3. [Binomial](#section3)
# 4. [Multinomial](#section4)
# 5. [Spike and slab](#section5) ?
#
#
# *References*
#
# [[1]](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) Integration from scipy [[2]](https://stackoverflow.com/questions/66502401/how-to-integrate-normal-distribution-with-numpy-and-scipy) Stackoverflow
#
# Note that: *Integrate* is much computationally consuming therefore impelementation of analytical expressions for some priors have been used in the package instead.
#
# ---

# ### **Imports**

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
from scipy import integrate
from math import log
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

# BC package imports TODO: do not use * imports, but be specific 
from bayesian_clustering.data_simulation import initialize_true
from bayesian_clustering.priors import g_normcard, g_multicard, check_sums
from bayesian_clustering.card_based import fill_cluster, fill_card, binom_op
from bayesian_clustering.prediction_post import partition_to1d_labels, partition_undo1d
from bayesian_clustering.distances import perfect_match_distance, variation_of_information

# %load_ext autoreload
# %autoreload 2
# -

# ## **Theory**
# ---
# $$\forall i, \quad \quad P_{X,\mathbf{I},i}(x_i) = \int P_{\theta_i}(x_i) d\Pi_{\mathbf{I},i}(\theta_i) = \int P_{\theta_i}(x_i) \Pi_{\mathbf{I},i}(\theta_i) d\theta_i$$
# $$  x_i \sim \mathcal{N}(\theta_i,1), \quad \quad \theta_i \sim G(\theta_i) $$
# Therefore the density functions for $x_i$ and $\theta_i$,
# $$P_{\theta_i}(x_i) = \phi(x_i, \theta_i, 1) \quad \text{ and } \quad \Pi_{\mathbf{I},i}(\theta_i) = g(\theta_i)$$

# We assume the data come from the normal distribution and we further pick a prior for the model. The priors we consider are,
#
# - Normal
# - Binomial
# - Multinomial
# - Spike and slab

# ### **Normal prior**
# ---
#
# $$x_i \sim \mathcal{N}(\theta_i,1), \quad \quad  \theta_i \sim \mathcal{N}(\mu_i,1)$$
# $$P_{X,\mathbf{I},i}(x_i) = \int \phi(x_i, \theta_i, 1) g(\theta_i) d\theta_i  =  \int \phi(x_i, \theta_i, 1)\, \phi(\theta_i, \mu_i, 1) d\theta_i\, , \quad \quad \forall i $$
# where $\mu_i$ is fixed in advance.
# - previous implementation

def g_card_anal(x_i: float, mu: float) -> float:
    normal_comp  = sps.norm.pdf(x_i)
    exp_comp = np.exp((x_i**2 + 2*mu*x_i-mu**2)/4)
    return normal_comp*exp_comp*np.sqrt(1./2)


g_card_anal(2.5,3)


# - updated version

def normal_conv(x_i: float, mu:float) -> float:
    value = integrate.quad(lambda theta: sps.norm.pdf(x_i, theta, 1)* sps.norm.pdf(theta, mu, 1),-np.inf,+np.inf)
    return value[0]


normal_conv(2.5, 3)


# ### **Gamma**
# ---
#
# $$\forall i\,, \quad \quad x_i \sim \mathcal{N}(\theta_i,1), \quad \quad  \theta_i \sim Gamma(a)$$
# $$P_{X,\mathbf{I},i}(x_i) = \int \phi(x_i, \theta_i, 1)\, g(\theta_i, a) d\theta_i\,$$
# $$g(\theta_i,a) = \frac{x^{a-1}e^{-\theta_i}}{\Gamma(a)}$$
#
# $\small \theta_i \in [0,\inf]$

def gamma_prior(x_i: float, a:float) -> float:
    value = integrate.quad(lambda theta: sps.norm.pdf(x_i, theta, 1)* sps.gamma.pdf(theta, a),-np.inf,+np.inf)
    return value[0]


gamma_prior(2.5,0.5)


# ##  *Cardinality-based*

# ### **Normal prior**

# $$m\sim \mathcal{N}(0,1)$$

def g_card(m_size:int,  K:int, sigma:int =1,)-> list:
    dist = sps.norm(loc=0, scale=sigma)
    if K == 2:
        m_size = m_size-1
    x = np.linspace(
        dist.ppf(0.001), dist.ppf(0.999), num=m_size
    )  # chooses n points between the 0.001 quantile and the 0.999 quantile.
    prior_g = dist.pdf(x) / dist.pdf(x).sum()
    return list(prior_g)


# ### **Binomial prior** [[3]](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)

# We put the binomial prior in the following way:
#
# In k = 2 groups, $m\in [n]$.
# $$m \sim Bin(n,p)$$
# Then $$f(m) = \binom{n}{m} p^{m}(1-p)^{n-m}$$

def binom_card(n:int, m:list[int], p:float) -> float:
    dist = sps.binom(n,p)
    values = [dist.pmf(k) for k in m]
    values = values/sum(values)
    return values


binom_card(10, range(1,11), 0.4)



# ### **Multinomial prior** [[4]](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html)

# $$\mathbf{m} \sim Multinomial(n,\mathbf{p})$$
# $$f(x_1, x_2, ..., x_k) = \frac{n!}{x_1!x_2!...x_k!} p_1^{x_1} p_2^{x_2} ... p_k^{x_k}$$

def multi_card(n:int, m:list[tuple], p:list[float]) -> float:
    dist = sps.multinomial(n,p)
    v = {k: dist.pmf(list(k)) for k in m}
    values = {vs: v[vs]/sum(v.values()) for vs in v.keys()}
    return values


multi_card(5, [(1,3,1),(2,1,2)], [0.3,0.6, 0.1])



# ## Distances

# - [[Paper]]("https://pdf.sciencedirectassets.com/272481/1-s2.0-S0047259X07X02343/1-s2.0-S0047259X06002016/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDrD0QHQDSEUjW0VP%2Fqv5e6d97ukVZlbdwb9%2F7QcX5pzAiAWRXLu1TP1usA0ON%2FD01bi%2FBRjhk8RR5LkqgaanFcHUCqyBQgWEAUaDDA1OTAwMzU0Njg2NSIMiz%2BaQnOeI9Dji9V%2BKo8FahH3PLpO%2BZBjYiadINafdsR%2Bl7CiBAIsaQ9xvisBpbrOYAwSTZmpeKkm5dmA1MSbdz5HTYX0W97pD1N8MmGCgA2mQ1JdZRPLxbKP4zrtM997LrZNjhk%2BvtHA5XH3w9Gs3eqmmOO7irUxXcwlL39warj26BpKccSczVwzImmeZq3FodyDNZkNCEfNRKISx%2F3acaxTs2tBcjmX0q4aWUQQMGbTYbBwGfSZGpxUXKL%2FDZqRdpA2iFBciarVpR5BRnMAOwgXvZlb9r23rz5g091wbkOQrQJo16kxFzKtevznOeMc57McpdlbgIOAM4yWWjAjj3vo14KanKrvx2GkqAi2pzjYNbu6ub0JOUl6pCYgZTswythvY0rdfHmgPqrb4jEDPWRAd%2FQMpge%2BnW6S3CPSqrMPJ6WFbJqZoEje1o7gll04WmF0ukV%2BEKJcLz%2FGXEodSCWSEtvLSWrI3veZ1pFEfoaHkAaLQ1gPu8c6z9yYRNtbwUK%2FhKmMNpamqevY7KKkXrmQ1CGortGUnpwzXyp8LZEoMExZ2YZ9jD5k79dSl6valbvIsjoaBWSUzRpH%2F4XyuNKCYS1aNo9yR6QPt%2Fio8DszFmRMDg0z1znKmdp83OAhNS7rAkgIPKptY1Rfvo7JaLj3Veyq56jZELDBGxece%2FaujcTTdVA7EFJkMKU58vEhlH%2BFZSkLNxjRP99RbDUKJOO8L45EJSgVQeYzMD%2FkMGWjVagWdi8Oe%2BYUCnj31mkWu%2BINjOaKqUlah2ZEWnRgUK%2BjWJUy%2FRDgEcahqlt2c%2FA67EDU%2Bwt2LakrGqRtxHCYkc%2BhNS1GLMpbURT0q8Z8Cu%2B3%2F88L%2BTtF7Z8nU3gBKT7Ae8gonBk1WmnLwTWyhjDnyLe9BjqyAV74L8eppAWwi%2FylGGglxiKT4IP4kB6jqGnGP7DjGRg2IV3tgKzhl455bH6yGlEf2R2neB9WhGCvuNMGq%2F041X77QBV87sNMdt5SA19Zp3Wny91cN94ZDfhpOAVSWA5yqfoFw2oPF%2BDd0UeJcF38ZD0Kpg4gpZV7wS4892PK7A91jPRZav0h7X1vetNieRpzodBGn%2ByaB2DpAPYkxJUmyVswiJY%2BsSsapuwnnAI5EwbJ9%2BM%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250213T124243Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSTZILLVV%2F20250213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=39672b0d6e286d66eaa5a11199538496eacbf2867bb13c5cc0316b6764911a8f&hash=cadaa305f517ba0d942ea120f92406dc8f748b54e9d54534636697b737466ce8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0047259X06002016&tid=spdf-aa0fa413-851e-46ac-9101-b0d733f8a98a&sid=a34663ca840f364d529b3f545506c0413c2egxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=140a5f5353540150505451&rr=9114e6620ece0a68&cc=nl")
#
#   - [Distances]("https://stats.stackexchange.com/questions/24961/comparing-clusterings-rand-index-vs-variation-of-information")
#    - [VIF]("https://clusteringjl.readthedocs.io/en/latest/varinfo.html")

# ### 1. [[Variation of information]]("https://en.wikipedia.org/wiki/Variation_of_information#:~:text=In%20probability%20theory%20and%20information,expression%20involving%20the%20mutual%20information.")
# $$VI(X,Y) = \sum_{i,j} r_{ij} [log(r_{ij}/p_{i})+log(r_{ij}/p_j)]$$
#
# Entropy of partition X,  $\quad H(X) = -\sum_i p_i log \,p_i$
# $$VI(X,Y) = 2 H(X\wedge Y) - H(X) -H(Y)$$

# +

# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
#
# https://en.wikipedia.org/wiki/Variation_of_information


# +
#VI <= 2*log(max(#clusters) and VI<= log(n)
# -

# Identical partitions
X1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
Y1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
X1, Y1 = partition_to1d_labels(10, X1).tolist(), partition_to1d_labels(10, Y1).tolist()
print("VI", variation_of_information(X1, Y1)[0], "1-VI", variation_of_information(X1, Y1)[1], "VI-adj", variation_of_information(X1, Y1)[2])
print("VI should be bounded by # clusters", 2*log(2), "number of points", log(10)) 
#1-VI = 1, VI = 0, 

# Similar partitions
X2 = [ [1,2,3,4], [5,6,7,8,9,10] ]
Y2 = [ [1,2,3,4,5,6], [7,8,9,10] ]
X2, Y2 = partition_to1d_labels(10, X2).tolist(), partition_to1d_labels(10, Y2).tolist()
print(variation_of_information(X2, Y2))
# VI = 1.102


# +
# Totally different partitions
X4 = [ [1,2,3,4,5,6,7,8,9,10] ]
Y4 = [ [1], [2], [3], [4], [5], [6], [7], [8], [9], [10] ]
X4, Y4 = partition_to1d_labels(10, X4).tolist(), partition_to1d_labels(10, Y4).tolist()
print(variation_of_information(X4, Y4)[:])
# VI = 2.30 (maximum VI is log(N) = log(10) = 2.30)
log(10), 2*log(10), variation_of_information(X4, Y4)[0]/(2*log(10))

#This is not very accurate since K* that one devides with should be the maximum number of clusters that minize the VI
#In my case because I will varying K, I scale with K(max_true, predicted)
# -

# https://gist.github.com/jwcarr/626cbc80e0006b526688

# ### 2. [[Rand index]]("https://en.wikipedia.org/wiki/Rand_index")
# $$RI = \frac{TP+TN}{TP+FP+FN+TN}$$
# the Rand index represents the frequency of occurrence of agreements over the total pairs, or the probability that 
# X and Y will agree on a randomly chosen pair

# ### 3. Adjusted Rand index
# $$ARI = \frac{RI - \mathbb{E}( RI)}{max(RI) - \mathbb{E}( RI)}$$
# [sklearn doc]("https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html")

from sklean.metrics import rand_score, adjusted_rand_score

# ### - split and join dinstance [[1]]("https://igraph.org/python/api/latest/igraph.clustering.html")
#
# van Dongen D: Performance criteria for graph clustering and Markov cluster experiments. Technical Report INS-R0012, National Research Institute for Mathematics and Computer Science in the Netherlands, Amsterdam, May 2000.

# !pip install igraph

from igraph import split_join_distance, compare_communities

X2 = [0,0,0,0,1,1,1,1,1,1]
Y2 = [0,0,0,0,0,0,1,1,1,1]
split_join_distance(X2, Y2, False)

X3 = [1,0,0,2,1,0,1,0,1,1]
Y3 = [0,0,1,0,0,0,1,1,1,1]
print(split_join_distance(X3, Y3, False))

compare_communities(X2,X3)

# ### Split and Join 
# - *Github - distances* [here]("https://github.com/micans/mcl/tree/main/src")
# - [page]("https://micans.org/mcl/")

# ### [[Check this github]]("https://escherba.github.io/clustering-metrics/clustering_metrics.metrics.html")

# ### [[Extra]]("https://www.geeksforgeeks.org/clustering-metrics/")

# ### Distances: Hamming
# $$d(A,B) = |A\Delta B| = |A \textbackslash B| + |B\textbackslash A| = r(A\cup B) - r(A\cap B)$$
#  Scipy: is simply the proportion of disagreeing components in u and v.
#  $$ \frac{c_{10}+c_{01}}{n}$$
#  $c_{01}$:number of element in A and not in B.
#  
#  $c_{10}$:number of element in B and not in A.

from scipy.spatial import distance

true_1d = [2, 0, 0, 0, 1, 0, 1, 1, 1, 2]
est_1d = [2, 0, 0, 0, 2, 0, 2, 2, 2, 2]

distance.hamming(true_1d, est_1d)

# ### Distances: min-cost perfect matching 
# - [reference](https://cstheory.stackexchange.com/questions/6569/edit-distance-between-two-partitions)
# - [wiki](https://en.wikipedia.org/wiki/Assignment_problem)
# - Hungarian Algorithm
#
#   $$D(P,Q) = min\{|A^c|:\emptyset\subset A \subseteq N, P^A = Q^A\}$$
#

distance = perfect_match_distance(est_1d, true_1d)
print(f"Partition-Distance D(P, Q) = {distance}")

from bayesian_clustering.distances import variation_of_information
from math import log
vi_res = variation_of_information(est_1d, true_1d)[0]
(vi_res/log(len(est_1d)))*100, '%'








