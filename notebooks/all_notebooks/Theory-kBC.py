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
#     display_name: Python [conda env:base]
#     language: python
#     name: conda-base-py
# ---

# ## k-groups Bayesian Clustering
# ---
# This notebook will serve as expanding the 2-group notebook to k-groups prior on n observations both in cardinality-based and product form cases.
#
# **Table of content** 
#
# 0. [General case](#general-case)
# 1. [Cardinality](#case-1)
#    <details> <summary><i> subsections </i></summary>
#        
#    - 1.1 [Main formulas](#main-formulas-1)
#    - 1.2 [Normal case](#norm-1)
#    - 1.3 [Implementation](#impl-1)</details>
# 2. [Product form](#case-2)
#    <details> <summary><i> subsections </i></summary>
#        
#    - 2.1 [ Main formulas](#main-formulas-2)
#    - 2.2 [Normal case](#norm-2)
#    - 2.3 [Implementation](#impl-2)
# </details>
#
# **Questions to be addressed**
# 1. *What is a convolution of two densities?*
# 2. return operation or return variable preferable?
#    
# **TBD**
# - how many groups of size $\mathbf{m}$ are there?
#     - specify according to functional

import numpy as np
import sympy as sp
import random
import scipy.stats as sps
from itertools import combinations
from itertools import groupby
import more_itertools as mit

# <a id="general-case"></a>
# ---
# # **0. General case:**
# ---

# The model,
#
# $$X_i \sim \mathcal{N}(\theta_i, \sigma^2), \quad \forall i \in [n] $$
# $$ \theta_i | \mathbf{I} \sim \Pi_{\mathbf{I},i} \quad \quad \theta \sim \sum_{\mathbf{I} \in \mathcal{I}_k }\lambda_{\mathbf{I}} \Pi_{\mathbf{I}} $$
# $$ \mathbf{I} \sim \lambda_{\mathbf{I}},\quad \quad \sum_{\mathbf{I} \in \mathcal{I}_k} \lambda_{\mathbf{I}} = 1$$

# ### **0.1 Main formulas**
# ---

#
# The **prior**,
# $$ \Pi(\mathbf{\theta}) = \sum_{I^*\in \mathcal{I}_k} P(\mathbf{\theta} | \mathbf{I} = I^*) P(\mathbf{I}=I^*) = \sum_{I^*\in \mathcal{I}_k} P(\mathbf{\theta}| \mathbf{I} = I^*) \lambda_{\mathbf{I}}  = \sum_{I\in \mathcal{I}_k} \lambda_{\mathbf{I}} \Pi_{\mathbf{I}}  $$
#
# The **Marginal distribution**,
#
# $$P_X = P(X = x) = \sum_{I \in \mathcal{I}_k} \lambda_{\mathbf{I}} P_{X,\mathbf{I}} = \sum_{I \in \mathcal{I}_k} \lambda_{\mathbf{I}} \bigotimes_{i\in[n]}  P_{X,\mathbf{I},i}(x_i) = \sum_{I \in \mathcal{I}_k} \lambda_{\mathbf{I}} \bigotimes_{l\in[k]} \bigotimes_{i \in I_l} P_{X,\mathbf{I}_l,i}(x_i)$$
#
# The **posterior $\Pi(\mathbf{I}|X)$**,
#
# $$\Pi(\mathbf{I}|X) = \frac{\lambda_{\mathbf{I}}P_{X,\mathbf{I}}}{P_X} = \frac{\lambda_{\mathbf{I}}\bigotimes_{i \in [n]}P_{X,\mathbf{I},i}(x_i)}{P_X} =\frac{\lambda_{\mathbf{I}}\bigotimes_{l \in [k]}\bigotimes_{i\in I_l} P_{X,\mathbf{I}_l,i}(x_i)}{P_X}   $$
#
# The **posterior $\Pi(\theta|X)$**
#
# $$ \Pi(\theta|X) = \sum_{\mathbf{I}\in \mathcal{I}_k} \Pi_{\mathbf{I}}(\theta|X) \Pi(\mathbf{I}|X) = \sum_{\mathbf{I}\in \mathcal{I}_k} \lambda_{\mathbf{I}} \bigotimes_{l \in [k]}  \bigotimes_{i \in I_l} \Pi_{I_l,i} (\theta_i | X_i)\frac{\bigotimes_{l \in [k]}\bigotimes_{i\in I_l} P_{X,I_l,i}(x_i)}{P_X} =  \sum_{\mathbf{I}\in \mathcal{I}_k} \lambda_{\mathbf{I}} \frac{\bigotimes_{l \in [k]}\bigotimes_{i\in I_l}\Pi_{I_l,i} (\theta_i | X_i)\times P_{X,I_l,i}(x_i)}{P_X}  $$
# $$\text{Using, } \quad \quad \quad \Pi_{I_l, i}(\theta_i | X_i) = \frac{\Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) }{P_{X, I_l, i}(X_i)}$$
#
# **$$ \Pi(\theta|X) = \sum_{\mathbf{I}\in \mathcal{I}_k} \lambda_{\mathbf{I}} \frac{\bigotimes_{l \in [k]}\bigotimes_{i\in I_l} \Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) }{P_X}$$**

# <a id="case-1"></a>
# ---
# # **1. Cardinality based $g(\mathbf{m})$**
#
# ---

#
#
# The cardinality prior $g(\mathbf{m}) = g(m_1,..,m_{k})$,
#
# $$\lambda_{\mathbf{m}} = c_{\mathbf{m}} g(\mathbf{m}) \quad \quad \textit{ for $\mathbf{m}\in \mathcal{M}$ and $g(\cdot)$ non-negative}$$
#
# The core assumption of this setting,
#
# $$\sum_{\mathbf{I}\in \mathcal{I}_k} \lambda_{\mathbf{I}} = 1$$
#
# <11details><summary><b> Case 1</b></summary>
# $$1= \sum_{\mathbf{I}\in \mathcal{I}_k} \lambda_{I} =\sum_{\mathbf{m}\in \mathcal{M}} c_{\mathbf{m}} g(\mathbf{m}) \quad\quad \mathcal{M} = \{ (m_1,..,m_k): \sum_k m_j =n \}  $$
#     
# For a specified cardinality $\mathbf{m}$, there are,$$ \binom{n}{m_1}\cdot \binom{n-m_1}{m_2} \cdots \binom{n-\sum_{l \in [k-2]} m_l}{m_{k-1}}$$
# excluding the last cluster which is going to be assigned the rest of the observations.
#
# It must then hold that, 
#
# $$c_{\mathbf{m}} = \Big[\binom{n}{m_1}\cdot \binom{n-m_1}{m_2} \cdots \binom{n-\sum_{l \in [k-2]} m_l}{m_{k-1}}\Big]^{-1} = \binom{n}{m_1,m_2,...,m_k}^{-1}$$ 
#
# So that the prior,
#
# $$\sum_{\mathbf{m} \in \mathcal{M}} g(\mathbf{m}) = 1 $$
# </details>
#
# The functional in this case is going to be,
#
# $$ \mathbf{Z} = (Z_1,..,Z_k)  \to \prod_{i \in [n]}[\psi_1(x_i)Z_1 + \psi_{2}(x_i)Z_2+\dots+\psi_{k}(x_i)Z_{k}]$$
# where $Z_1 = 1$.
#
# From this we will need the $\mathbf{m}$- coefficient based on the cardinalities.

# ---
#
# ### For k=3
# $\mathbf{m} = ( m_1, m_2, m_3)$ where $m_1+m_2+m_3 = n$
#
# The ways to choose $m_1,m_2,m_3$ is like the known **[[stars and bars]](https://brilliant.org/wiki/identical-objects-into-distinct-bins/#:~:text=arrangements%20are%20there?-,General%20Case%20of%20Identical%20Objects%20into%20Distinct%20Groups,are%20chosen%20for%20the%20bars.)** combinatorial example with the difference of exluding 0 for each of the coordinate.
#
# Found on stach overflow **[[initializing without 0]](https://math.stackexchange.com/questions/3540823/stars-and-bars-distribution-with-without-empty-cells)**
#
# This means there are $$ \binom{n-1}{k-1}$$ for choosing cardinality $\mathbf{m}$
#
# For each $\mathbf{m}$ there are $$\binom{n}{m_1}\binom{n-m_1}{m_2}\binom{n-m_1-m_2}{m_3}$$
# note the last term is $\binom{m_3}{m_3} = 1$.

# <a id="main-formulas-1"></a>
#
# ### **1.1 Main formulas**
#
# ---

# - The **marginal**,
# $$P_X = P(X = x) = \sum_{\mathbf{I} \in \mathcal{I}} \lambda_{\mathbf{I}} P_{X,\mathbf{I}} =\sum_{\mathbf{I} \in \mathcal{I}_k} \lambda_{\mathbf{I}} \bigotimes_{l\in[k]} \bigotimes_{i \in I_l} P_{X,I_l,i}(x_i) = \sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}}\Big( g(\mathbf{m}) \bigotimes_{l\in[k]} \bigotimes_{i \in I_l} P_{X,I_l,i}(x_i) \Big) $$
# $$ P_X = \sum_{\mathbf{m} \in \mathcal{M}} g(\mathbf{m}) A_{\mathbf{m}}(P_{X,I_1}(x),P_{X,I_2}(x),\dots, P_{X,I_k}(x))$$
#
# - The **prior**,
#
#   $$ \Pi =\sum_{\mathbf{I} \in \mathcal{I}} \lambda_{\mathbf{I}} \Pi_{\mathbf{I}} =\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}}\Big( g(\mathbf{m}) \bigotimes_{l\in[k]} \bigotimes_{i \in I_l} \Pi_{I_l,i}(x_i) \Big) = \sum_{\mathbf{m} \in \mathcal{M}} g(\mathbf{m}) A_{\mathbf{m}}(\Pi_{I_1}(x),\Pi_{I_2}(x),\dots, \Pi_{I_k}(x)) $$
# - The **posterior** $\Pi(\mathbf{I}|X)$
#
# $$\Pi(\mathbf{I}|X)=\frac{\lambda_{\mathbf{I}} P_{X,\mathbf{I}}}{P_X}=\frac{\lambda_{\mathbf{I}} \bigotimes_{l\in[k]} \bigotimes_{i \in I_l} P_{X,I_l,i}(x_i)}{ P_X}$$
#
#   
# - The **posterior** $\Pi(\mathbf{\theta}|X)$,
# $$\Pi(\theta|X) = \sum_{\mathbf{I}\in \mathcal{I}_k} \lambda_{\mathbf{I}} \frac{\bigotimes_{l \in [k]}\bigotimes_{i\in I_l} \Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) }{P_X} $$
#
#
# Finally,
# $$\Pi(\theta|X) = \frac{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m}) A_{\mathbf{m}}(\Pi_{I_1}(\theta) P_{\theta}(x),\,\Pi_{I_2}(\theta) P_{\theta}(x),\dots, \,\Pi_{I_k}(\theta)P_{\theta}(x))}{\sum_{\mathbf{m} \in \mathcal{M}} g(\mathbf{m}) A_{\mathbf{m}}(P_{X,I_1}(x),P_{X,I_2}(x),\dots, P_{X,I_k}(x))} = \frac{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m}) A_{\mathbf{m}}(\rho_1(x),\rho_2(x),\dots \rho_k(x))}{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m})A_{\mathbf{m}}(P_{X,I_1}(x), P_{X,I_2}(x),\dots,P_{X,I_k}(x))} \quad \text { with } \quad \rho_l = \Pi_{I_l}(\theta) P_{\theta}(x), \quad {\small l \in [k]}.$$
#
# - The **component posterior** $p_{l,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)$,
# $$p_{l*,i} = \sum_{\mathbf{I}: i\in I_l} \lambda_{\mathbf{I} } \frac{\bigotimes_{l\in[k]} \bigotimes_{i \in I_l} P_{X,I_l,i}(x_i)}{P_X} = \frac{\sum_{m\in \mathcal{M}} \sum_{\substack{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}\\ i \in I_{l*}}} g(\mathbf{m})  \bigotimes_{l\in[k]} \bigotimes_{j \in I_l} P_{X,I_l,j}(x_j)}{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m})A_{\mathbf{m}}(P_{X,I_1}(x), P_{X,I_2}(x),\dots,P_{X,I_k}(x))} = \frac{P_{X,I_{l*},i}(x_i)\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m})  \bigotimes_{l\in[k]} \bigotimes_{\substack{j \in I_l \\ j\neq i}} P_{X,I_l,j}(x_j)}{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m})A_{\mathbf{m}}(P_{X,I_1}(x), P_{X,I_2}(x),\dots,P_{X,I_k}(x))}$$
#
# $$p_{l*,i} =  \frac{P_{X,I_{l*},i}(x_i)}{P_X}\sum_{m\in \mathcal{M}}  g(\mathbf{m})  A_{\mathbf{m}_{l*},-i}(P_{X,I_1}(x), P_{X,I_2}(x), \dots, P_{X,I_k}(x)), \quad \quad \text{ for } l* \in [k].$$
# with, 
# $$\mathbf{m}_{l*} = \begin{cases} (m_1,\dots,\; m_{l*}-1,\dots,m_{k}), \quad \text{ for } l* \in [k-1] \\ \mathbf{m} \quad \quad l* = k \end{cases}$$

# <a id="norm-1"></a>
#
# ### **1.2 Normal case** 
# ---

# - The **marginal**,
# $$ P_X = \sum_{\mathbf{m} \in \mathcal{M}} g(\mathbf{m}) A_{\mathbf{m}}\Big[\mathcal{N}(0,\sigma^2)*\Pi_{I_1}(x),\;\mathcal{N}(0,\sigma^2)*\Pi_{I_2}(x_i),\;\dots,\; \mathcal{N}(0,\sigma^2)*\Pi_{I_k}(x_i)\Big]$$
#
# - The **prior**,
#   $$ \Pi = \sum_{\mathbf{m} \in \mathcal{M}} g(\mathbf{m}) A_{\mathbf{m}}(\Pi_{I_1}(x),\Pi_{I_2}(x),\dots, \Pi_{I_k}(x)) $$
#
# - The **posterior** $\Pi(\mathbf{I}|X)$
#
# $$\Pi(\mathbf{I}|X)=\frac{\lambda_{\mathbf{I}} P_{X,\mathbf{I}}}{P_X}=\frac{\lambda_{\mathbf{I}} \bigotimes_{l\in[k]} \bigotimes_{i \in I_l}\mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i)}{ \mathcal{N}(0,\sigma^2)*\Pi_{\mathbf{I},i}(x_i)}$$
#
#   
# - The **posterior** $\Pi(\mathbf{\theta}|X)$,
#
# $$\Pi(\theta|X) = \frac{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m}) A_{\mathbf{m}}(\rho_1(x),\rho_2(x),\dots \rho_k(x))}{\sum_{m\in \mathcal{M}} \sum_{\mathbf{I}:\, c(\mathbf{I})= \mathbf{m}} g(\mathbf{m})A_{\mathbf{m}}(\mathcal{N}(0,\sigma^2)*\Pi_{I_1}(x),\; \mathcal{N}(0,\sigma^2)*\Pi_{I_2}(x),\dots,\mathcal{N}(0,\sigma^2)*\Pi_{I_k}(x))} \quad \text { with } \quad \rho_l = \Pi_{I_l}(\theta) \phi(x,\theta,\sigma^2), \quad {\small l \in [k]}.$$
#
# - The **component posterior** $p_{l,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)$,
# $$p_{l*,i} =  \frac{\mathcal{N}(0,\sigma^2)*\Pi_{I_{l*},i}(x_i)\sum_{m\in \mathcal{M}}  g(\mathbf{m})  A_{\mathbf{m}_{l*},-i}(\mathcal{N}(0,\sigma^2)*\Pi_{I_1}(x),\; \mathcal{N}(0,\sigma^2)*\Pi_{I_2}(x),\dots,\mathcal{N}(0,\sigma^2)*\Pi_{I_k}(x))}{P_X}, \quad \quad \text{ for } l* \in [k].$$
# with, 
# $$\mathbf{m}_{l*} = \begin{cases} (m_1,\dots,\; m_{l*}-1,\dots,m_{k}), \quad \text{ for } l* \in [k-1] \\ \mathbf{m}, \quad \quad l* = k\,. \end{cases}$$
#
# We replace the marginal density as $P_{X,I_l,i}(x_i) = \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i)$ and $P_{\theta_i}(x_i) = \phi(x_i, \theta_i, \sigma^2)$.

# <a id="impl-1"></a>
#
# ### **1.3 Implementation**
#
# ---

# - Create all possible cardinalities $\mathbf{m}$ which are in total $\binom{n}{m_1 m_2...m_k}= \frac{n!}{m_1!m_2!...m_k!}$. !!
#
# Say $\mathcal{M} = \{\mathbf{m}\; | \sum_j^k m_j = N \}$
# - Draw $\mathbf{M}$ from them according to the **prior** $g(\mathbf{m})$
#   
# From all possible partitions with cardinality vector $\mathbf{M}$,
# - Draw uniformly Partition $\mathbf{I}$.
#   
# ---

# ** The example below uses [more_itertools]("https://more-itertools.readthedocs.io/en/stable/api.html")

# +
np.random.seed(101)
N = 5
K = 3

#Find all possible combinations of cardinalities
def combos_m(sample_size:int, cluster_size:int)->(list,list):
    def backtrack(target, k, start, current, result):
        if k==0 and target==0:
            result.append(list(current))
            return
        if k<0 or target<0: 
            return
        for i in range(1, target+1):
            current.append(i)
            backtrack(target-i, k-1,1, current, result)
            current.pop()
    result = []
    backtrack(sample_size, cluster_size, 1, [], result)
    result_s = [''.join(str(x) for x in card) for card in result]
    return result, result_s
    
def g_card(m_size:int, sigma:int =1)-> list: #Poisson distribution, Binomial
    dist = sps.norm(loc=0, scale=sigma)
    x = np.linspace(
        dist.ppf(0.001), dist.ppf(0.999), num=m_size
    )  # chooses n points between the 0.001 quantile and the 0.999 quantile.
    prior_g = dist.pdf(x) / dist.pdf(x).sum()
    return prior_g
    
def get_prior_per_m(prior_values:list, all_ms:list)->dict:
    #ms_to_str = [''.join(str(x) for x in card) for card in all_ms]
    prior_dict = dict.fromkeys(all_ms)
    for idx, m in enumerate(all_ms):
        prior_dict[m] = prior_values[idx] 
    return prior_dict
    
def pick_cardinality(prior:list, all_ms: list)->str:
    m = np.random.choice(all_ms, p = prior)
    return m
    
all_nk_list, all_nk = combos_m(N,K); print(all_nk)
g_prior = g_card(len(all_nk))
prior_dict = get_prior_per_m(g_prior, all_nk)
M = pick_cardinality(g_prior, all_nk)
print(f"We pick cardinality {M} with probability {prior_dict[M]} out of all possible cardinalities")


# +
def all_partitions(n:int, k:int)->list:
    data = [i for i in range(1, n+1)]
    all_sets = list(mit.set_partitions(data,k)) #not order preserving [1][2][3 4] = [3 4][1][2]
    return all_sets

def partitions_m(all_parts:list, ms:list)->list:
    parts_m = []
    ms.sort()
    for partition in all_parts:
        flag = True
        partition = sorted(partition,key=len)
        for idx_m, m in enumerate(ms):
            if len(partition[idx_m])!=m:
                flag = False
                continue
        if flag:
            parts_m.append(partition)
    return parts_m
    
def choose_subset_m(all_parts_m):
    out_set_idx = np.random.choice(range(len(all_parts_m)))
    return out_set_idx, all_parts_m[out_set_idx]
    
partition_space = all_partitions(N,K)
m_list = [int(i) for i in M]; print(m_list, M)
sets_m = partitions_m(partition_space, m_list); 
idx_subset, picked_subset = choose_subset_m(sets_m)
print(f"We pick subset {picked_subset} with index{idx_subset}, with probability {round(1/len(sets_m),4)},  uniformly from all subsets of cardinality {M}:\n{sets_m} ")
print(f"The probability of picking this particular subset from the partition space is, prior_g(m) * 1/#(sets of card m) which evaluates to{round(prior_dict[M]/len(sets_m), 4)}")


# -

# ## The functional

def mult_polynomial(priors_list:list, arg: int = None) -> list:
    k =  len(priors_list) #[1,2,3]
    var = np.array([1] + list(sp.symbols(f'z1:{k}'))) ; print(var)
    mat_mon = np.matrix(priors_list); print(mat_mon)
    prod_all = var@mat_mon; print(prod_all)
    if arg != None :
        prod_all = np.delete(prod_all, arg); print(prod_all)
    prod_poly = np.prod(var@mat_mon)
    return prod_poly, prod_poly.as_poly()
mult_polynomial([[1,4],[2,3], [0,0]])

# <a id="case-2"></a>
# ---
# # **2. Product form for $\lambda_{\mathbf{I}}$**
# ---

# ### **2.0 The model**
# ---

# $$X_i \sim \mathcal{N}(\theta_i, \sigma^2), \quad \forall i \in [n] $$
# $$ \theta_i | \mathbf{I} \sim \Pi_{\mathbf{I},i} \quad \quad \theta \sim \sum_{\mathbf{I} \in \mathcal{I}_k }\lambda_{\mathbf{I}} \Pi_{\mathbf{I}} $$
# $$\theta_i \sim \sum_l^{k-1}\omega_{l,i} \Pi_{l,i}+(1-\sum_{l}^{k-1}\omega_{l,i})\Pi_{k,i}$$
# $$ \mathbf{I} \sim \lambda_{\mathbf{I}},  \quad \quad \sum_{\mathbf{I} \in \mathcal{I}_k} \lambda_{\mathbf{I}} = 1$$

#
# In this case , $$\lambda_{\mathbf{I}} = C_{\lambda}  \prod_{i_1 \in I_1} \lambda_{1,i_1} \cdots \prod_{i_{k-1}\in I_{k-1}} \lambda_{k-1,i_{k-1}}\,\, \text{ or }\,\, \lambda_{\mathbf{I}} = C_{\lambda}  \prod_{l \in [k-1]}\prod_{i\in I_l} \lambda_{l,i} \quad \text{ with } \quad C_{\lambda} = \prod_{i = 1}^{n} (\lambda_{1,i}+ \dots + \lambda_{k-1,i} +1)^{-1}$$ 
#
# <details><summary><i></i> Click here for details </summary>
# The functional again,
# $$\prod_{i=1}^{n} a_{1,i}+a_{2,i}+ \dots a_{k-1,i} = \sum_{\mathbf{I} \in \mathcal{I}_k} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k-1} \in I_{k-1}} a_{k-1,i_{k-1}}$$
#
# In advance we set,
#     $$ \Lambda = \begin{bmatrix}
# \lambda_{1,1}& \cdots & \lambda_{1,n-1} & \lambda_{1,n} \\
# \lambda_{2,1} & \cdots & \lambda_{2,n-1} & \lambda_{2,n} \\
# \vdots & \ddots & \ddots & \vdots \\
# \lambda_{k,1} & \cdots & \cdots & \lambda_{k,n}
# \end{bmatrix}$$
# </details>
#
#

# <a id="main-formulas-2"></a>
#
# ### **2.1 Main formulas**
# ---

# <a id="norm-2"></a>
#
# - The **marginal**,
# $$ P_X =  \sum_{\mathbf{I} \in \mathcal{I}_k}\Big( C_{\lambda} \cdot \lambda_{\mathbf{I}} \bigotimes_{l \in [k]}\bigotimes_{i\in I_l} P_{X,I_l,i}(x_i)\Big) =  \sum_{\mathbf{I} \in \mathcal{I}_k}\Big( C_{\lambda} \prod_{l\in [k-1]}\prod_{i\in I_l}\lambda_{l,i} \bigotimes_{l \in [k]}\bigotimes_{i\in I_l} P_{X,I_l,i}(x_i)\Big) = C_\lambda \sum_{\mathbf{I} \in \mathcal{I}_k} \Big(\bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l}\lambda_{l,i} P_{X,I_l,i}(x_i)\times \bigotimes_{j \in I_k} P_{X,I_k,j}(x_j)\Big)  $$
#
# $$ P_X = C_{\lambda} \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \lambda_{l,i} P_{X,I_l,i}(x_i) + P_{X,I_k,i}(x_i)\Big] = \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i) \Big] \quad \text{ where }\quad \omega_{l,i}= \frac{\lambda_{l,i}}{\sum_{s=1}^{k-1} \lambda_{s,i}+1}, \quad\small{l \in [k-1], \; i \in [n]}.$$
#
# - The **prior**,
#
#   $$ \Pi = C_{\lambda} \sum_{I \in I_k} \lambda_{\mathbf{I}} \bigotimes_{l \in [k]}\bigotimes_{i\in I_l} \Pi_{I_l,i} = C_\lambda \sum_{I \in I_k} \Big(\bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l}\lambda_{l,i} \Pi_{I_l,i}(x_i)\times \bigotimes_{j \in I_k} \Pi_{I_k,j}(x_j)\Big) = C_{\lambda} \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \lambda_{l,i} \Pi_{I_l,i}(x_i) + \Pi_{I_k,i}(x_i)\Big] = \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,i} \Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) \Pi_{I_k,i}(x_i)\Big]$$
#
# - The **posterior** $\Pi(\mathbf{I}|X)$
#
#   $$\Pi(\mathbf{I}|X)=\frac{\lambda_I P_{X,\mathbf{I}}}{P_X} =  \frac{C_{\lambda}  \prod_{l \in [k-1]}\prod_{i\in I_l} \lambda_{l,i} P_{X,\mathbf{I}}}{P_X} =  \frac{C_\lambda \bigotimes_{l \in [k-1]} \bigotimes_{i\in I_l}\lambda_{l,i} P_{X,I_l,i}(x_i)\times \bigotimes_{j \in I_k} P_{X,I_k,j}(x_j)}{P_X}$$
#
#   
# - The **posterior** $\Pi(\mathbf{\theta}|X)$,
# $$\Pi(\mathbf{\theta}|X) = \sum_{I\in\mathcal{I}} \Pi_I(\theta|X) \Pi(I|X) = \sum_{\mathbf{I}\in \mathcal{I}_k}\lambda_{\mathbf{I}} \frac{\bigotimes_{l \in [k]}\bigotimes_{i\in I_l} \Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) }{P_X} = \frac{\sum_{\mathbf{I} \in \mathcal{I}_k}C_{\lambda}\Big[ \bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l} \lambda_{l,i} \Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) \times  \bigotimes_{j \in I_k} \Pi_{I_k, j}(\theta_j) \cdot P_{\theta_j}(X_j)\Big] }{  \sum_{\mathbf{I} \in \mathcal{I}_k} C_\lambda \Big[\bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l}\lambda_{l,i} P_{X,I_l,i}(x_i)\times \bigotimes_{j \in I_k} P_{X,I_k,j}(x_j)\Big] }$$
#
# $$\Pi(\theta|X) = \bigotimes_{i\in [n]}\frac{\sum_{l \in [k-1]} \omega_{l,i}\Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big)\Pi_{I_k, i}(\theta_i) \cdot P_{\theta_i}(X_i)\Big]}{\Big[\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)\Big]}$$
#
# - The **component posterior** $p_{l,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)$,
# $$ p_{l*,i} = \frac{\omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} \quad \text{ for } l* \in [k-1] \quad \text{ and } \quad p_{k,i} = \frac{(1- \sum_{l \in [k-1]} \omega_{s,i})\cdot P_{X,I_{k},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} $$
# <details><summary><i> Click here for details </i></summary>
#  $$p_{l*,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)  = \sum_{\mathbf{I}: i\in I_{l*}}  \frac{C_\lambda \Big[\bigotimes_{l \in [k-1]}  \bigotimes_{j\in I_l}\lambda_{l,j} P_{X,I_l,j}(x_j)\times \bigotimes_{j' \in I_k} P_{X,I_k,j'}(x_{j'})\Big]}{P_X} = P_{X,I_{l*},i}(x_{i})\frac{\lambda_{l,i}}{(\sum_{s=1}^{k-1}\lambda_{s,i}+1)}\sum_{\mathbf{I}: i\in I_l} C_{\lambda_{\mathbf{I}},-i }\frac{ \bigotimes_{l \in [k-1]}  \bigotimes_{j\in I_l, j\neq i}\lambda_{l,j} P_{X,I_l,j}(x_j)\times \bigotimes_{j' \in I_k} P_{X,I_k,j'}(x_{j'})}{P_X}$$
# $$ p_{l*,i} = \omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})\frac{\bigotimes_{j\in [n], j\neq i}\Big[\sum_{l \in [k-1]} \omega_{l,j} P_{X,I_l,j}(x_j) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,j} \Big) P_{X,I_k,j}(x_j)\Big]}{\bigotimes_{j\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,j} P_{X,I_l,j}(x_j) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,j} \Big) P_{X,I_k,j}(x_j)\Big]} = \frac{\omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} \quad \text{ for } l* \in [k-1]$$  
# Similarly for the $k^{th}$ clustering,
# $$ p_{k,i} = \frac{(1- \sum_{l \in [k-1]} \omega_{s,i})\cdot P_{X,I_{k},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} $$
# The functional again,   
# $$\prod_{i=1}^{n} a_{1,i}+a_{2,i}+ \dots a_{k,i} = \sum_{\mathbf{I} \in \mathcal{I}_k} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k} \in I_k} a_{k,i_{k}}$$
# Suppose $\mathcal{I}_{i*} = \{ \mathbf{I} \in \mathcal{I}_k: i* \in I_l\}$
# $$\sum_{\mathbf{I} \in \mathcal{I}_{i*}} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k} \in I_{k}} a_{k,i_{k}} = a_{l,i*}\sum_{\mathbf{I} \in \mathcal{I}_{i*}\backslash \{i*\}} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k-1} \in I_{k-1}} a_{k-1,i_{k-1}} = \prod_{i=1}^{n-1} a_{1,i}+a_{2,i}+ \dots a_{k,i} $$
# </details>

# <a id="norm-2"></a>
#
# - The **marginal**,
# $$ P_X =  \sum_{\mathbf{I} \in \mathcal{I}_k}\Big( C_{\lambda} \cdot \lambda_{\mathbf{I}} \bigotimes_{l \in [k]}\bigotimes_{i\in I_l} P_{X,I_l,i}(x_i)\Big) =  \sum_{\mathbf{I} \in \mathcal{I}_k}\Big( C_{\lambda} \prod_{l\in [k-1]}\prod_{i\in I_l}\lambda_{l,i} \bigotimes_{l \in [k]}\bigotimes_{i\in I_l} P_{X,I_l,i}(x_i)\Big) = C_\lambda \sum_{\mathbf{I} \in \mathcal{I}_k} \Big(\bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l}\lambda_{l,i} P_{X,I_l,i}(x_i)\times \bigotimes_{j \in I_k} P_{X,I_k,j}(x_j)\Big)  $$
#
# $$ P_X = C_{\lambda} \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \lambda_{l,i} P_{X,I_l,i}(x_i) + P_{X,I_k,i}(x_i)\Big] = \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i) \Big] \quad \text{ where }\quad \omega_{l,i}= \frac{\lambda_{l,i}}{\sum_{s=1}^{k-1} \lambda_{s,i}+1}, \quad\small{l \in [k-1], \; i \in [n]}.$$
#
# - The **prior**,
#
#   $$ \Pi = C_{\lambda} \sum_{I \in I_k} \lambda_{\mathbf{I}} \bigotimes_{l \in [k]}\bigotimes_{i\in I_l} \Pi_{I_l,i} = C_\lambda \sum_{I \in I_k} \Big(\bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l}\lambda_{l,i} \Pi_{I_l,i}(x_i)\times \bigotimes_{j \in I_k} \Pi_{I_k,j}(x_j)\Big) = C_{\lambda} \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \lambda_{l,i} \Pi_{I_l,i}(x_i) + \Pi_{I_k,i}(x_i)\Big] = \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,i} \Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) \Pi_{I_k,i}(x_i)\Big]$$
#
# - The **posterior** $\Pi(\mathbf{I}|X)$
#
#   $$\Pi(\mathbf{I}|X)=\frac{\lambda_I P_{X,\mathbf{I}}}{P_X} =  \frac{C_{\lambda}  \prod_{l \in [k-1]}\prod_{i\in I_l} \lambda_{l,i} P_{X,\mathbf{I}}}{P_X} =  \frac{C_\lambda \bigotimes_{l \in [k-1]} \bigotimes_{i\in I_l}\lambda_{l,i} P_{X,I_l,i}(x_i)\times \bigotimes_{j \in I_k} P_{X,I_k,j}(x_j)}{P_X}$$
#
#   
# - The **posterior** $\Pi(\mathbf{\theta}|X)$,
# $$\Pi(\mathbf{\theta}|X) = \sum_{I\in\mathcal{I}} \Pi_I(\theta|X) \Pi(I|X) = \sum_{\mathbf{I}\in \mathcal{I}_k}\lambda_{\mathbf{I}} \frac{\bigotimes_{l \in [k]}\bigotimes_{i\in I_l} \Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) }{P_X} = \frac{\sum_{\mathbf{I} \in \mathcal{I}_k}C_{\lambda}\Big[ \bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l} \lambda_{l,i} \Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) \times  \bigotimes_{j \in I_k} \Pi_{I_k, j}(\theta_j) \cdot P_{\theta_j}(X_j)\Big] }{  \sum_{\mathbf{I} \in \mathcal{I}_k} C_\lambda \Big[\bigotimes_{l \in [k-1]}\bigotimes_{i\in I_l}\lambda_{l,i} P_{X,I_l,i}(x_i)\times \bigotimes_{j \in I_k} P_{X,I_k,j}(x_j)\Big] }$$
#
# $$\Pi(\theta|X) = \bigotimes_{i\in [n]}\frac{\sum_{l \in [k-1]} \omega_{l,i}\Pi_{I_l, i}(\theta_i) \cdot P_{\theta_i}(X_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big)\Pi_{I_k, i}(\theta_i) \cdot P_{\theta_i}(X_i)\Big]}{\Big[\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)\Big]}$$
#
# - The **component posterior** $p_{l,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)$,
# $$ p_{l*,i} = \frac{\omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} \quad \text{ for } l* \in [k-1] \quad \text{ and } \quad p_{k,i} = \frac{(1- \sum_{l \in [k-1]} \omega_{s,i})\cdot P_{X,I_{k},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} $$
# <details><summary><i> Click here for details </i></summary>
#  $$p_{l*,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)  = \sum_{\mathbf{I}: i\in I_{l*}}  \frac{C_\lambda \Big[\bigotimes_{l \in [k-1]}  \bigotimes_{j\in I_l}\lambda_{l,j} P_{X,I_l,j}(x_j)\times \bigotimes_{j' \in I_k} P_{X,I_k,j'}(x_{j'})\Big]}{P_X} = P_{X,I_{l*},i}(x_{i})\frac{\lambda_{l,i}}{(\sum_{s=1}^{k-1}\lambda_{s,i}+1)}\sum_{\mathbf{I}: i\in I_l} C_{\lambda_{\mathbf{I}},-i }\frac{ \bigotimes_{l \in [k-1]}  \bigotimes_{j\in I_l, j\neq i}\lambda_{l,j} P_{X,I_l,j}(x_j)\times \bigotimes_{j' \in I_k} P_{X,I_k,j'}(x_{j'})}{P_X}$$
# $$ p_{l*,i} = \omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})\frac{\bigotimes_{j\in [n], j\neq i}\Big[\sum_{l \in [k-1]} \omega_{l,j} P_{X,I_l,j}(x_j) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,j} \Big) P_{X,I_k,j}(x_j)\Big]}{\bigotimes_{j\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,j} P_{X,I_l,j}(x_j) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,j} \Big) P_{X,I_k,j}(x_j)\Big]} = \frac{\omega_{l*,i} \cdot P_{X,I_{l*},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} \quad \text{ for } l* \in [k-1]$$  
# Similarly for the $k^{th}$ clustering,
# $$ p_{k,i} = \frac{(1- \sum_{l \in [k-1]} \omega_{s,i})\cdot P_{X,I_{k},i}(x_{i})}{\sum_{l \in [k-1]} \omega_{l,i} P_{X,I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) P_{X,I_k,i}(x_i)} $$
# The functional again,   
# $$\prod_{i=1}^{n} a_{1,i}+a_{2,i}+ \dots a_{k,i} = \sum_{\mathbf{I} \in \mathcal{I}_k} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k} \in I_k} a_{k,i_{k}}$$
# Suppose $\mathcal{I}_{i*} = \{ \mathbf{I} \in \mathcal{I}_k: i* \in I_l\}$
# $$\sum_{\mathbf{I} \in \mathcal{I}_{i*}} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k} \in I_{k}} a_{k,i_{k}} = a_{l,i*}\sum_{\mathbf{I} \in \mathcal{I}_{i*}\backslash \{i*\}} \prod_{i_1 \in I_1} a_{1,i_1}\prod_{i_2 \in I_2} a_{2,i_2} \cdots \prod_{i_{k-1} \in I_{k-1}} a_{k-1,i_{k-1}} = \prod_{i=1}^{n-1} a_{1,i}+a_{2,i}+ \dots a_{k,i} $$
# </details>

# <a id="norm-2"></a>
#
# ### **2.2 Normal case**
# ---

# + [markdown] jp-MarkdownHeadingCollapsed=true
# $$ P_{X,I_l,i} = \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i)$$
#
# - The **marginal**,
# $$ P_X =  \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,i}\, \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) \mathcal{N}(0,\sigma^2)*\Pi_{I_k,i}(x_i) \Big] = \quad \text{ where }\quad \omega_{l,i}= \frac{\lambda_{l,i}}{\sum_{s=1}^{k-1} \lambda_{s,i}+1}, \quad\small{l \in [k-1], \; i \in [n]}.$$
#
# - The **posterior** $\Pi(\mathbf{I}|X)$
#
#   $$\Pi(\mathbf{I}|X) =  \frac{C_\lambda \bigotimes_{l \in [k-1]} \bigotimes_{i\in I_l}\lambda_{l,i} \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i)\times \bigotimes_{j \in I_k} \mathcal{N}(0,\sigma^2)*\Pi_{I_k,i}(x_i)}{P_X}$$
#
#   
# - The **posterior** $\Pi(\mathbf{\theta}|X)$,
# $$\Pi(\theta|X) = \bigotimes_{i\in [n]}\Big\{\frac{\sum_{l \in [k-1]} \omega_{l,i}\Pi_{I_l, i}(\theta_i) \cdot \phi(X_i,\theta_i,\sigma^2) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big)\Pi_{I_k, i}(\theta_i) \cdot \phi(X_i,\theta_i,\sigma^2)}{\sum_{l \in [k-1]} \omega_{l,i} \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big)\mathcal{N}(0,\sigma^2)*\Pi_{I_k,i}(x_i)}\Big\}$$
#
# - The **component posterior** $p_{l,i} = \sum_{\mathbf{I}: i \in I_l}\Pi(\mathbf{I}|X)$,
# $$ p_{l*,i} = \frac{\omega_{l*,i} \cdot \mathcal{N}(0,\sigma^2)*\Pi_{I_{l*},i}(x_i)}{\sum_{l \in [k-1]} \omega_{l,i} \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) \mathcal{N}(0,\sigma^2)*\Pi_{I_k,i}(x_i)} \quad \text{ for } l* \in [k-1] \quad \text{ and } \quad p_{k,i} = \frac{(1- \sum_{l \in [k-1]} \omega_{s,i})\cdot \mathcal{N}(0,\sigma^2)*\Pi_{I_k,i}(x_i)}{\sum_{l \in [k-1]} \omega_{l,i}\, \mathcal{N}(0,\sigma^2)*\Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) \mathcal{N}(0,\sigma^2)*\Pi_{I_k,i}(x_i)} $$
#
# <details><summary><i> Click here for details </i></summary>
# - The prior stays the same,
#     $$ \Pi = \bigotimes_{i\in [n]}\Big[\sum_{l \in [k-1]} \omega_{l,i} \Pi_{I_l,i}(x_i) + \Big( 1 - \sum_{l \in [k-1]}\omega_{l,i} \Big) \Pi_{I_k,i}(x_i)\Big]$$
# </details>
#
# --- 
# -
# ## **Simulation dataset** - example:  [[2]]("https://statsthinking21.github.io/statsthinking21-python/10-BayesianStatistics.html")
#

#

#





