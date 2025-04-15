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

# ## Introduction to Bayesian Clustering
# ---
#
# The aim of this notebook is to formulate the bayesian clustering setting and core formulas that will later be used for analysing and implementing the main part of the project.
#
# __Table of contents__
# 1. [Preliminaries](#preliminaries)
# 2. [Known partition of size k](#knowni)
# 3. [Main Formulas for known partition](#mainknown)
# 4. [Unknown partition of k groups](#unknownkgroups)
# 5. [Main formulas for k=2: two approaches](#2unknown)
# 6. [Component posterior](#comp-post)
# 7. [Normal-case data](#norm-data)
#
# __Questions to be addressed__
# - Q1: **Question: How many subsets are there for 2 groups and how many for k groups?**
# - Q2: **HOW are the two ways (card-based, prod-form) connected?**
# - Q3: [Convolution between distributions](#q-norm)
#
# __To be done__
# 1. [Normal data cace + Plot](#norm-case)
# ---

from itertools import combinations
import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt

# <a id="preliminaries"></a>
# $$
# \require{color}
# $$
#
# # **0. Preliminaries [[1]](https://arxiv.org/abs/1904.01003) [[2]](./papers/k-group_prior_draft_for_Ellie.pdf)**
#
# ---

# - ### The dataset
# $$X = (X_1,..,X_n) \;\; s.t. \;\; X_i \sim \mathcal{N}(\theta_i, \sigma^2)$$
# and thus the vector $$\mathbf{\theta} = (\theta_1,..,\theta_n) \in \mathbb{R}^n$$
#
# - ### Linear subspaces
# In the clustering setting say, $\mathbf{I} = (I_1,.., I_k)$ where $I_i$ is the $i^{th}$ cluster containing the set of coordinates for cluster i.
# The Linear subspaces $\mathbb{L}_{\mathbf{I}}$ for $\theta$,
# $$
# \mathbb{L}_{\mathbf{I}} = \left\{
# \mathbf{\theta} \in \mathbb{R}^n : \theta_j = \theta_{j'}, \forall j, j'\in I_i, \; i \in [k]
# \right\} $$
#
# <a id="knowni"></a>
# ## Known partition $\mathbf{I}$ of size k
# The model is,
#
# $$X_i \sim \mathcal{N}(\theta_i, \sigma^2), \quad \forall i \in [n] $$
# $$ \theta_i \sim \Pi_{I_{l},i} \quad \forall l \in k, \forall i \in I_l$$

# #### Example <a id='another_cell'></a>
# Note - When you run the above c
# Suppose k = 2 and $\mathbf{X} = (X_1,X_2,..,X_{6}), \quad X_i\sim \mathcal{N}(\theta_i,\sigma^2) \quad $ for $i\in 1,..,7$.
# $$\mathbf{I} = (I_1,I_2,I_3), \quad I_1= \{1,3,5\},\; I_2 = \{2,4\}\; \text{ and }\; I_3 = \{6, 7\}$$
#
# The vector $\mathbf{\theta} = (\theta_1,\theta_2,\theta_1,\theta_2,\theta_1,\theta_3, \theta_3)$ meaning that the pairs of observations $(X_1,X_3,X_5)$, $(X_2,X_4)$ and $(X_6, X_7)$ belong to different clusters and are grouped together by $\theta$, repspectively.
#
# The prior on $\theta$,
#
# $$\theta_1 \sim \Pi_{I_{1},1} \quad \quad \quad \theta_2 \sim \Pi_{I_{2},1}$$
# $$\theta_3 \sim \Pi_{I_{1},3} \quad \quad \quad \theta_4 \sim \Pi_{I_{2},4}$$
# $$\theta_5 \sim \Pi_{I_{1},5} \quad \quad \quad \theta_6 \sim \Pi_{I_{3},6}$$
# $$\theta_7 \sim \Pi_{I_{3},7}$$
#

# +
I1 = np.random.normal(loc=2, scale=1.0, size=3)
I2 = np.random.normal(5, scale=1, size=2)
I3 = np.random.normal(10, scale=1, size=2)
print(I1)
data = {"partition_I": ["I1", "I2", "I3"], "points": [I1, I2, I3]}

df = pd.DataFrame(data)
dI = df.explode("points").reset_index(drop=True)
dI["points"] = dI.points.astype(float)
dI


# +
fig, ax = plt.subplots(figsize=(8, 5))
Partitions = ["I1", "I2", "I3"]

vp = pt.half_violinplot(
    x="points",
    y="partition_I",
    inner=None,
    data=dI,
    width=1.5,
    ax=ax,
)

# Iterate over the species
for i, specI in enumerate(Partitions):
    # Subset the data
    data = dI[dI["partition_I"] == specI]

    # Jitter the values on the vertical axis
    y = i + np.random.uniform(high=0.1, size=len(data))

    # Select the values of the horizontal axis
    x = data["points"]

    # Add the rain using the scatter method.
    ax.scatter(x, y, alpha=0.6)

plt.title(r"Sample with a known partition $I$")


# -

# <a id="mainknown"></a>
#
# # 1. **Unknown Partition: Main formulas**
# ---

# #### Prior on $\theta$
#
# The prior can be rewritten as,
# $$ \Pi_{\mathbf{I}} = \bigotimes_{i\in[n]} \Pi_{\mathbf{I},i} = \bigotimes_{l\in[k]}\bigotimes_{i\in I_l} \Pi_{I_l,i}$$
#
# #### Similarly, the marginal distribution,
#
# $$ P_{X, \mathbf{I}} =  \bigotimes_{i \in [n]} P_{\,X,\,\mathbf{I},\,i}\,(x_i) = \bigotimes_{l \in [k]}\bigotimes_{i \in I_l} P_{\,X,\,I_l,\,i}\,(x_i) $$
#
# Using the above, $P_{X,I,i}(x_i) = \int P_{\theta_i}(x_i) \, d\Pi_{I,i}(\theta_i)=\int P_{\theta_i}(x_i) \, d\Pi_{I_l,i}(\theta_i)$
#
# #### The posterior
# $$\Pi_{\mathbf{I}} (\theta \mid X) = \bigotimes_{i \in [n]} \Pi_{\mathbf{I}, i} (\theta_i \mid X_i) = \bigotimes_{l \in [k]} \bigotimes_{i \in I_l} \Pi_{I_l, i} (\theta_i \mid X_i)$$
#
# where,
# $$\Pi_{\mathbf{I}, i}(\theta_i \mid X_i) = \frac{\Pi_{\mathbf{I}, i}(\theta_i) P_{\theta_i}(X_i)}{P_{X, \mathbf{I}, i}(X_i)},
# \quad \Pi_{I_l, i}(\theta_i \mid X_i) = \frac{\Pi_{I_l, i}(\theta_i) P_{\theta_i}(X_i)}{P_{X, I_l, i}(X_i)}
# \text{if} \, i \in I_l, \, l \in [k]
# $$

# ## *Example k=2, uknown partition*
# ---

# Suppose the *partition space*, $\mathcal{I}_2 = \mathcal{I}$ and the *clusters*, $I_1 = I$ and $ I_2=I^c$.
#
# This mean that, it is the same to pick $I_1$ and let the rest of the obs to belong to $I_2$, therefore there is no **product form** for $\lambda_I$
# $$\lambda_{\mathbf{I}} =  \lambda_I$$
#
# (as we are sincere about group two and $\lambda_{I^c} = 1$)

# ### 1. $\lambda_I$ depending on group cardinality
#
# Using the above, one can argue that $\lambda$ depends on the cardinality of group $I$ in the following way,
#
# $$\lambda_{I} = c_{|I|} g(|I|), \quad \quad \text{ for } I \in \mathcal{I} \text{ and } g(\cdot) \textit{ non-negative}$$
#
# - $g(\cdot)$ spreads mass according to the cardinality
# - $\lambda_I$ spreads mass according to $I$
# $$\pi_n(m) \propto g(m) \binom{n}{m}, \quad \quad m\in [n]$$
#
# <a id="questions-card"></a>
# <details> <summary><i> Click here for detailed proof </i></summary>
# The assumption we lead by is,
#     $$\sum_{I \in \mathcal{I}} \lambda_I =1$$
# Say that $\lambda_I = c_{|I|}\, g(|I|)$,
#     $$1 = \sum_{I \in \mathcal{I}} \lambda_I =\sum_{m \in [n]} \sum_{I: |I| =m} c_m\cdot g(m) =  \sum_{m\in [n]} \binom{n}{m} \; c_m\cdot g(m) $$
#
# We pick $$c_m =  \binom{n}{m}^{-1}, \quad \sum_{m \in [n]} g(m) = 1\quad \text{ and the prior}\quad \pi_n(m)= c_m\;g(m) \binom{n}{m} = g(m), \quad \quad m\in [n]$$
# </details>

# ### 1.1 Implementation [[1]](https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-4/Needles-and-Straw-in-a-Haystack--Posterior-concentration-for/10.1214/12-AOS1029.full)
#
# We are using the functional for quantity $Q_n$:
#
# $Q_n = \sum_{|S| = m}
# \prod_{i \in S} \psi(X_i)
# \prod_{i \notin S} \phi(X_i)$
# is equal to the coefficient of $Z^m$ in the polynomial
# $$Z\to \prod_{i\in [n]}( \phi(x_i)+\psi(x_i)Z)$$
#
# We denote $A_m(\cdot, \cdot)$ the $m^{th}$ coefficient of the above mentioned polynomial and $A_{m-1, -i}(\cdot, \cdot)$ the $(m-1)$-order coefficient when $i^{th}$ element is excluded.
# <details><summary>The details</summary>
# $$p_z(a,b) = \prod_{j\in [n]}( a(x_j)+b(x_j)z) = A_0(a,b)+A_1(a,b)z^1+\dots+A_n(a,b)z^n$$
# $$p_{z,-i}(a,b) = \prod_{j\in [n] \\ j\neq i}( a(x_j)+b(x_j)z) =\frac{p_z(a,b)}{((a(x_i)+b(x_i)z)} =   A_{0,-i}(a,b)+A_{1,-i}(a,b)z^1+\dots+A_{n-1,-i}(a,b)z^{n-1}$$
# </details>
#
# In order to compute the given product, for example $P_X =  \sum_{m=0}^{n} \sum_{I: |I|=m} g(m) \Big[ \bigotimes_{i\in I}  P_{X,\mathbf{I},i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,\mathbf{I},j}(x_j)\Big]$ we are using the known functional:
#
# $$\sum_{I: |I|=m}\Big[ \bigotimes_{i\in I}  P_{X,\mathbf{I},i}(x_i)\times \bigotimes_{j\in I^c}  P_{X,\mathbf{I},j}(x_j)\Big] =A_m(P_{X,1}(x), P_{X,2}(x))$$
# where we set,
# $$\phi(\cdot) = P_{X,1}(\cdot)\quad \text{ and } \quad \psi(\cdot) = P_{X,2}(\cdot)$$
# $$\phi(X) = (P_{X,1}(x_1),., P_{X,1}(x_n)) , \quad \text{ and } \psi(X) = (P_{X,2}(x_1),., P_{X,2}(x_n)) , \quad$$
#
# **-!!-Meaning that we can implement the above computation, using a package in quadratic number of operations or $nlog(n)$ by clever one.**


# #### Remarks
# - $ [[1,4][2,3]] \not\equiv [[2,3][1,4]]$
#     - for example, say n = 4, then the polynomial coefficient we want for m = 2 would be,
#       $$(a_1 a_2 b_3 b_4 + a_1 b_2 b_3 a_4 + b_1 a_2 a_3 b_4 + a_1 b_2 a_3 b_4 + b_1 a_2 b_3 a_4 + b_1 b_2 a_3 a_4)$$
# - Trivial case to consider the empty set for I. We ignore the 0 and n degree coefficient of the polynomial.

# +
# Polynomial of degree len(a)
def mult_polynomial(a: list, b: list, *args) -> list:
    """

    Parameters
    ----------
    a: coefficient referring to the 2nd group
    b: coefficient referring to the 1st group
    args: only a number in case ith element should be excluded

    Returns
    -------
    p(z) = prod_{i in [n]} (a_i z + b_i), Polynomial of degree n
    """
    p_ab = np.poly1d(1)
    n = len(a)
    for idx in range(n):
        if idx in args:
            continue
        p_ab *= np.poly1d([a[idx], b[idx]], variable="z")  # a_i z + b_i
    return p_ab


def get_am(p: list, m: int) -> float:
    """

    Parameters
    ----------
    p: polynomial
    m: power of monomial

    Returns
    -------
    p[m]: coefficient of m-th monomial of polynomial p
    """
    if m not in range(len(p) + 1):
        return -6977.64
    else:
        return p[m]


polyp = mult_polynomial(
    np.random.normal(1,10,10),
    np.random.normal(5,5, 10),
)
print(polyp)
get_am(polyp, 1)  # get coefficient of degree 1
## I need to plug in the densities P_x,

# -

print(np.poly1d([1, 2, 3]))

# <a id="implement"></a>
#
# ### **Implementation**
# ---

# #### **The prior $\lambda_I$ can always be modeled in two steps: first draw the random cardinality M according to the prior $\pi_n$, and then given M = m, draw a random set I uniformly (i.e., with probability $\binom{n}{m}^{-1}$ from the family of all subsets of [n] of cardinality m.**

# + endofcell="--"
N = 4


# g(m)
def prior_card(n: int, sigma: int = 1) -> list:
    """

    Parameters
    ----------
    sigma: is from Normal dist
    n: sample size

    Returns
    -------
    g_card: list of probabilities resembling the normal distribution

    """
    # define the normal distribution and PDF
    dist = sps.norm(loc=0, scale=sigma)
    x = np.linspace(
        dist.ppf(0.001), dist.ppf(0.999), num=n
    )  # chooses n points between the 0.001 quantile and the 0.999 quantile.
    prior_g = dist.pdf(x) / dist.pdf(x).sum()
    return prior_g


def cardinalities(n: int) -> list:
    """

    Parameters
    ----------
    n: sample size

    Returns
    -------
    all possible cardinalities [1,..,n]

    """
    return [card for card in range(1, n + 1)]


def pick_m(card: list, prior_g: list) -> int:
    """

    Parameters
    ----------
    card: list of possible cardinalities
    prior_g: list of probabilities assigned to each cardinality

    Returns
    -------
    a randomly picked cardinality based on prior_g
    """
    return np.random.choice(card, p=prior_g)


def choose_subset(card: list, m: int) -> (list, int):
    """
    card: cardinalities {1,..,n}
    m: specified cardinality
    Returns
    -------
    list of subsets of cardinality m, index of picked set from subset
    """
    all_subs_m = np.array(list(combinations(card, m)))
    out_set_ind = np.random.choice(range(len(all_subs_m)))
    return all_subs_m, out_set_ind


g_card = prior_card(N); print(g_card)
cards = cardinalities(N)
M = pick_m(cards, g_card)
all_subs_M, Out_set_ind = choose_subset(cards, M)
print(M, all_subs_M)

print(
    f"Set {all_subs_M[Out_set_ind]} with index {Out_set_ind} is sampled with prob {g_card[M]/len(all_subs_M):.4f} and its cardinality is sampled with prob {len(all_subs_M[Out_set_ind])} is sampled with probability g(m)= {g_card[M]:.4f}"
)
# -
# --


# ### 2. $\lambda_I$ as a product form of obs-component $\lambda_i$'s
# ---

# Suppose $$\lambda_{I} = C_{\lambda} \prod_{i\in I} \lambda_i $$ with $C_{\lambda}$ to be a normalizing constant of the form,
# $C_{\lambda} = \prod_{i\in [n]}(1+\lambda_i)^{-1}$
#
# In this context we use the functional, $$\prod_{i\in [n]} (a_i + b_i) = \sum_{I \in \mathcal{I}} \Big(\prod_{i \in I}a_i\Big)\Big(\prod_{j\in I^c} b_j\Big)$$
#
# <details> <summary> Quick-proof</summary>
# We set $\lambda = C_{\lambda} \prod_{i \in I} \lambda_i$ or, equivalently, $\lambda_I = C_{\lambda} \prod_{i \in I} \lambda_i \prod_{j \in I^c} 1$ then by applying the functional, the sum will be,
# $$\sum_{I \in \mathcal{I}} \lambda_I = \sum_{I \in \mathcal{I}} C_{\lambda} \prod_{i \in I} \lambda_i \prod_{j \in I^c} 1 =  C_{\lambda} \prod_{i\in [n]} (\lambda_i + 1) $$
# It must hold,
# $$\sum_{I \in \mathcal{I}} \lambda_I =1 $$
# Thus, $$ C_{\lambda} = \prod_{i\in [n]} (\lambda_i + 1)^{-1}$$
# </details>
#
# **Remark:** $\lambda_i$'s are fixed in advance $\forall i \in [n]$.

# <a id="2unknown"></a>
# ##  **2. Main formulas k = 2 groups**
#
#  1. dependent on cardinality of cluster $I$
#  2. priors are in product-form
# ---

# The **prior**
#
#  - dependent on cardinality,
#  $$ \Pi =\sum_{I \in \mathcal{I}} \lambda_I \Pi_I = \sum_{m=0}^{n} \sum_{I: |I|=m} g(m) \Big( \bigotimes_{i\in I}\Pi_{1,i} \times \bigotimes_{j\in I^c}\Pi_{2,j}\Big) = \sum_{m=0}^{n} g(m) A_m(\Pi_{1}, \Pi_{2}) $$
#
#  - in priors product form,
#  $$ \Pi =\sum_{I \in \mathcal{I}} \lambda_I \Pi_I = \sum_{I \in \mathcal{I}} C_{\lambda} \prod_{i \in I} \lambda_i \Big[\bigotimes_{i \in I} \Pi_{1,i} \times \bigotimes_{j\in I^c} \Pi_{2,j}\Big] =  C_\lambda \sum_{I \in \mathcal{I}}\Big(\bigotimes_{i\in I} \lambda_i \Pi_{1,i} \times \bigotimes_{j \in \mathcal{I}^c}  \Pi_{2,j}\Big)$$
# <a id="component-theta"></a>
#      Using the functional,
#  $$\Pi = \prod_{i \in [n]} (1+\lambda_i)^{-1} \bigotimes_{i \in [n]} \Big[\lambda_i \Pi_{1,i} + \Pi_{2,i}\Big] =\bigotimes_{i\in [n]} \Big[\lambda_i (\lambda_i +1)^{-1} \Pi_{1,i} + (1+\lambda_i)^{-1}  \Pi_{2,i} \Big]= \bigotimes_{i \in [n]} \Big[\omega_i \Pi_{1,i} + (1 - \omega_i) \Pi_{2,i}\Big],  \quad \text{ with, }\, \omega_i = \frac{\lambda_i}{1+\lambda_i}, \quad \small{ i \in [n]} $$
#
# The **Marginal**
# - dependent on cardinality,
# $$P_X = P(X = x) = \sum_{I \in \mathcal{I}} \lambda_{\mathbf{I}} P_{X,\mathbf{I}} = \sum_{I \in \mathcal{I}}\Big( \lambda_{\mathbf{I}} \bigotimes_{i\in I}  P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,2,j}(x_j)\Big) = \sum_{m=0}^{n} \sum_{I: |I|=m}\Big( g(m) \bigotimes_{i\in I}  P_{X,\mathbf{I},i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,\mathbf{I},j}(x_j) \Big) $$
# $$ P_X = \sum_{m=0}^{n} g(m) A_m(P_{X,1}(x),P_{X,2}(x))$$
# <a id="functional"></a>
# - in priors product form,
# $$P_X = P(X = x) = \sum_{I \in \mathcal{I}} \lambda_{\mathbf{I}} P_{X,\mathbf{I}}(x) = \sum_{I \in \mathcal{I}}\Big(C_{\lambda} \Pi_{i \in I} \lambda_i \bigotimes_{i\in I}  P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,2,j}(x_j)\Big) =C_{\lambda}\sum_{I \in \mathcal{I}}  \Big(\bigotimes_{i\in I} \lambda_i P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,2,j}(x_j) \Big)$$
#
#     Using the functional,
#
# $$P_X = C_{\lambda} \bigotimes_{i\in [n]} \Big[\lambda_i P_{X,1,i}(x_i) +  P_{X,2,j}(x_j)\Big] =  \bigotimes_{i\in [n]} \Big[\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i)\Big] $$
#
# The **posterior** $\Pi(I|X)$
#  - dependent on cardinality,
#  $$\Pi(I|X)=\frac{\lambda_I P_{X,I}}{P_X}=\frac{\lambda_I \bigotimes_{i\in I}  P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,2,j}(x_j)}{P_X}$$
#  - in priors product form,
#  $$\Pi(I|X)=\frac{\lambda_I P_{X,I}}{P_X} =  \frac{C_{\lambda} \Pi_{i \in I}\lambda_i P_{X,I}}{P_X} = C_{\lambda}\Pi_{i\in I} \lambda_i \frac{\bigotimes_{i\in I}P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}P_{X,2,j}(x_j)}{P_X}=C_{\lambda} \frac{\bigotimes_{i\in I}\lambda_i P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}P_{X,2,j}(x_j)}{P_X} $$
#

# <a id="posterior"></a>
# ###  **$\Pi(\theta|X)$ posterior**
#
# ---
#
# $$\Pi(\theta|X) = \sum_{I\in\mathcal{I}} \Pi_I(\theta|X) \Pi(I|X) = \sum_{I\in\mathcal{I}} \lambda_I \bigotimes_{i\in I} \Pi_{1,i}(\theta_i|x_i) \times \bigotimes_{j \in I^c} \Pi_{2,j}(\theta_j|x_j) \Big[ \frac{\bigotimes_{i\in I}  P_{X,1,i}(x_i) \times \bigotimes_{j\in I^c}  P_{X,2,j}(x_j)}{P_X}\Big] $$
#
# $$\Rightarrow \Pi(\theta|X) = \sum_{i\in I}\lambda_I\frac{\Big(\bigotimes_{i\in I} \Pi_{1,i}(\theta_i|x_i) \times P_{X,1,i}(x_i) \Big) \times \Big(\bigotimes_{j \in I^c} \Pi_{2,j}(\theta_j|x_j)\times P_{X,2,j}(x_j) \Big) }{P_X}$$
#
# It holds that,
#
# $$ \Pi_{l,i}(\theta_i|x_i) = \frac{\Pi_{l,i}(\theta_i)P_{\theta_i}(X_i)}{P_{X,l,i}(x_i)} \quad \text{ for } l=1,2 .$$
#
# Working out the expression,
#
# $$ \Pi(\theta|X) = \sum_{i\in I}\lambda_I \frac{\bigotimes_{i\in I} \Pi_{1,i}(\theta_i) P_{\theta_i}(x_i) \times \bigotimes_{j \in I^c}\Pi_{2,j}(\theta_j) P_{\theta_j}(x_j) }{P_X}$$
#
# - dependent on cardinality,
#
# $$\Pi(\theta|X) = \frac{\sum_{m=0}^{n}\sum_{I:|I|=m} g(m)\bigotimes_{i\in I} \Pi_{1,i}(\theta_i) P_{\theta_i}(x_i)\times \bigotimes_{j \in I^c}\Pi_{1,j}(\theta_j) P_{\theta_j}(x_j)}{\sum_{m \in [n]} g(m)A_m(P_{X,1}(x), P_{X,2}(x)) } = \frac{\sum_{m=0}^{n}g(m) A_m(\rho_1(x),\rho_2(x))}{\sum_{m \in [n]} g(m)A_m(P_{X,1}(x), P_{X,2}(x))} \quad \text { with } \quad \rho_l = \Pi_{l,i}(\theta_i) P_{\theta_i}(x), \quad {\small l =1,2}.$$
# - in priors product form,
#
#  $$\Pi(\theta|X) = \sum_{I\in\mathcal{I}} \Pi_I(\theta|X) \Pi(I|X) =\frac{C_{\lambda}\sum_{I\in\mathcal{I}} \Big(\bigotimes_{i\in I} \lambda_i \Pi_{1,i}(\theta_i) P_{\theta_i}(x_i) \times \bigotimes_{j \in I^c}\Pi_{2,j}(\theta_j) P_{\theta_j}(x_j)\Big) }{\bigotimes_{i\in [n]} \Big[\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i) \Big]} $$
#
#      Using the functional,
#
#  $$ \Rightarrow \Pi(\theta|X) =  \bigotimes_{i\in [n]}\frac{ \omega_i \Pi_{1,i}(\theta_i) P_{\theta_i}(x_i) +(1-\omega_i)\Pi_{2,i}(\theta_i) P_{\theta_i}(x_i) }{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i}(x_i) }$$

# <a id="comp-post"></a>
# ### **Component Posterior**
#
# ---
#
# The component posterior can be interpreted as the probability of $i \in I$ given the observed sample. It is of significant interest as it "outputs"  probability of i belonging in a certain cluster in the observed setting.
#
# $$p_i = P(i \in I|X) = \sum_{I: i \in I} \frac{\lambda_I \cdot P_{X,I}}{P_X} = \sum_{I: i\in I} \lambda_I  \frac{\bigotimes_{j\in I} P_{X,1,j}(x_j) \times \bigotimes_{j'\in I^c} P_{X,2,j'}(x_{j'})}{P_X}$$
# - dependent on cardinality,
#
# $$p_{1,i}  =\frac{P_{X,1,i}(x_i)\sum_{m =0}^{n} \, g(m) A_{m-1,-i}(P_{X,1}(x),P_{X,2}(x))
# }{\sum_{m\in [n]}g(m)A_m(P_{X,1}(x), P_{X,2}(x))},  \quad \quad p_{2,i} = \frac{P_{X,2,i}(x_i)\sum_{m =0}^{n} \, g(m) A_{m,-i}(P_{X,1}(x),P_{X,2}(x))
# }{\sum_{m\in [n]}g(m)A_m(P_{X,1}(x), P_{X,2}(x))} $$
# <details> <summary><i> Click here for detailed proof </i></summary>
# $$p_i = \sum_{I: i\in I} \lambda_I  \frac{\bigotimes_{j\in I} P_{X,1,j}(x_j) \times \bigotimes_{j'\in I^c} P_{X,2,j'}(x_{j'})}{P_X} = \frac{\sum_{m =0}^{n} \,\;\sum_{I: |I|=m, i\in I} g(m)  \bigotimes_{j\in I} P_{X,1,j}(x_j) \times \bigotimes_{j'\in I^c} P_{X,2,j'}(x_{j'})}{\sum_{m\in [n]}g(m)A_m(P_{X,1}(x), P_{X,2}(x))}=\frac{P_{X,1,i}(x_i)\sum_{m =0}^{n} \, g(m) \;\sum_{I: |I|=m-1,} \bigotimes_{j\in I, j\neq i} P_{X,1,j}(x_j) \times \bigotimes_{j'\in I^c} P_{X,2,j'}(x_{j'})}{\sum_{m\in [n]}g(m)A_m(P_{X,1}(x), P_{X,2}(x))}$$
# </details>
#
# - in priors product form,
#
# $$p_{1,i} = \frac{\omega_i P_{X,1,i}(x_{i})}{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i) } \quad \text{ and } p_{2,i} = \frac{(1-\omega_i) P_{X,2,i}(x_{i})}{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i) } $$
#
# <details> <summary><i> Click here for detailed proof </i></summary><b>
# $$ p_{1,i} = \sum_{I: i \in I} C_{\lambda} \prod_{j \in I} \lambda_j \frac{\bigotimes_{j\in I} P_{X,1,j}(x_j) \times \bigotimes_{j'\in I^c} P_{X,2,j'}(x_{j'})}{P_X} = C_{\lambda} \sum_{I: i \in I}\frac{ \bigotimes_{j\in I} \lambda_j P_{X,1,j}(x_j) \times \bigotimes_{j'\in I^c} P_{X,2,j'}(x_j)}{P_X} $$
#
# The initial functional,
#
# $$ \prod_{j \in [n]}( a_i + b_i) = \sum_{I\in \mathcal{I}} \prod_{j \in I} a_j \prod_{j' \in I^c} b_{j'}$$
#
# is now going to be restricted for sets $\mathcal{I}_{i^*} = \{I \in \mathcal{I}: i^* \in I\}$,
#
# $$ \sum_{I \in I_{i^*}} \prod_{j \in I} a_i \prod_{j' \in I^c} b_j = a_{i^*} \sum_{I \in I_{i^*}\backslash\{i^*\}} \prod_{j\in I} a_j\prod_{j' \in I^c}  b_{j'} = a_{i^*} \prod_{\mathclap{\substack{j \in [n] \\ j\neq i}}}( a_j + b_j) $$
#
# ![image.png](attachment:6beee495-026e-493d-a3b1-fbeefb5b4935.png)
#
# $$\Rightarrow p_{1,i} =  C_{\lambda}P_{X,1,i}(x_i) \lambda_i \bigotimes_{\mathclap{\substack{j \in [n]\\j \neq i}}}\frac{\lambda_j P_{x,1,j}(x_j) + P_{x,2,j'}(x_{j'})}{P_X}= \omega_i P_{X,1,i}(x_{i})\times \frac{\bigotimes_{j \in [n]\backslash\{ i\}} \omega_j P_{x,1,j}(x_j) +(1-\omega_j) P_{X,2,j}(x_j)}{P_X} $$
#
# $$p_{1,i} = \omega_i P_{X,1,i}(x_{i})\times \frac{\bigotimes_{j \in [n]\backslash\{ i\}} \omega_j P_{x,1,j}(x_j) +(1-\omega_j) P_{X,2,j}(x_j)}{P_X} = \omega_i P_{X,1,i}(x_{i})\times \frac{\bigotimes_{j \in [n]\backslash\{ i\}} \omega_j P_{x,1,j}(x_j) +(1-\omega_j) P_{X,2,j}(x_j)}{\bigotimes_{j\in [n]} \omega_j P_{X,1,j}(x_j) + (1-\omega_j)  P_{X,2,j
# }(x_j) }$$
#
# From previous section,
# $$P_X  =  \bigotimes_{j\in [n]} \omega_j P_{X,1,j}(x_j) + (1-\omega_j)  P_{X,2,j
# }(x_j) $$
#
# The final expression,
#
# $$p_{1,i} = \frac{\omega_i P_{X,1,i}(x_{i})}{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i) }$$
# </b>
# </details>
#
# <a id="norm-data"></a>
#
# ## **3. Normal prior case**
# ---
#
# <a id="q-norm"></a>
# - The **marginal**,
#
#     1. dep. on cardinality, $$ P_X = \sum_{m=0}^{n} g(m) A_m(P_{X,1}(x),P_{X,2}(x))$$
#
#     2. priors product form, $$ P_X = \bigotimes_{i\in [n]} \Big[\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i)\Big] = \bigotimes_{i\in [n]} \Big[\omega_i \mathcal{N}(0,\sigma^2)*\Pi_{I,i}(x_i) + (1-\omega_i)  \mathcal{N}(0,\sigma^2)*\Pi_{I^c,i}(x_i)\Big] $$
#
#
# - The **posterior** $\Pi(\theta|X)$,
#
#     See the posterior formulas derivation [here](#posterior)
#
#     1. dep. on cardinality,
#     $$\Pi(\theta|X) = \frac{\sum_{m=0}^{n}g(m) A_m(\rho_1(x),\rho_2(x))}{\sum_{m \in [n]} g(m)A_m(P_{X,1}(x), P_{X,2}(x))} \quad \text { with } \quad \rho_l = \Big[\Pi_{l,i}(\theta_i) P_{\theta_i}(x) \Big]_{i\in[n]}, \quad {\small l =1,2}$$
#       $$\Pi(\theta|X) = \frac{\sum_{m=0}^{n}g(m) A_m(\rho_1(x),\rho_2(x))}{\sum_{m \in [n]} g(m)A_m(P_{X,1}(x), P_{X,2}(x))} $$
#   $$\quad \text { with } \quad \rho_l = \Big[\Pi_{l,i}(\theta_i) \phi(X_i,\theta_i,\sigma^2) \Big]_{i\in [n]},\quad\quad P_{X,l}(x) = \Big[\mathcal{N}(0,\sigma^2)*\Pi_{l,j}(x_j)\Big]_{j \in [n]} \quad {\small l =1,2}. $$
#
#   2. priors product form
#      $$ \Rightarrow \Pi(\theta|X) =  \bigotimes_{i\in [n]}\frac{ \omega_i \Pi_{1,i}(\theta_i) P_{\theta_i}(x_i) +(1-\omega_i)\Pi_{2,i}(\theta_i) P_{\theta_i}(x_i) }{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i}(x_i) } = \bigotimes_{i\in [n]}\frac{ \omega_i \Pi_{1,i}(\theta_i) \phi(X_i,\theta_i,\sigma^2) +(1-\omega_i)\Pi_{2,i}(\theta_i) \phi(X_i,\theta_i,\sigma^2)}{\omega_i \mathcal{N}(0,\sigma^2)*\Pi_{I,i}(x_i) + (1-\omega_i) \mathcal{N}(0,\sigma^2)*\Pi_{I^c,i}(x_i) }$$
#
# - **component posterior** $p_i = \sum_{I: i \in I_l}\Pi(I|X)$,
#     1. dep on cardinality
# $$p_{1,i}  =\frac{P_{X,1,i}(x_i)}{P_X}\sum_{m =0}^{n} \, g(m) A_{m-1,-i}(P_{X,1}(x),P_{X,2}(x))
# ,  \quad \quad p_{2,i} = \frac{P_{X,2,i}(x_i)}{P_X} \sum_{m =0}^{n} \, g(m) A_{m,-i}(P_{X,1}(x),P_{X,2}(x))
# $$
# $$ P_{X,l}(x) = \Big[\mathcal{N}(0,\sigma^2)*\Pi_{l,j}(x_j)\Big]_{j \in [n]} \quad {\small l =1,2}. $$
#     2. priors product form
#        $$p_{1,i} = \frac{\omega_i P_{X,1,i}(x_{i})}{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i) } \quad \text{ and } p_{2,i} = \frac{(1-\omega_i) P_{X,2,i}(x_{i})}{\omega_i P_{X,1,i}(x_i) + (1-\omega_i)  P_{X,2,i
# }(x_i) } $$
#
# $$p_{1,i} = \frac{\omega_i \mathcal{N}(0,\sigma^2)*\Pi_{I,i}(x_i)}{\omega_i \mathcal{N}(0,\sigma^2)*\Pi_{I,i}(x_i) + (1-\omega_i)  \mathcal{N}(0,\sigma^2)*\Pi_{I^c,i}(x_i) } \quad \text{ and } p_{2,i} = \frac{(1-\omega_i) \mathcal{N}(0,\sigma^2)* \Pi_{I^c,i}(x_i)}{\omega_i \mathcal{N}(0,\sigma^2)*\Pi_{I,i}(x_i) + (1-\omega_i) \mathcal{N}(0,\sigma^2)*\Pi_{I^c,i}(x_i)} $$
#
# <details><summary><i> Click here for details</i> </summary>
#
# - Assuming $X_i \sim \mathcal{N}(\theta_i, \sigma^2), \quad i \in [n]$, it holds that $P_{\theta_i}(x_i) = \phi(x_i, \theta_i, \sigma^2)$ where $\phi(\cdot)$ is the normal density function, $\theta_i$ is the mean and $\sigma$ the st. deviation.
#
# The component marginal is by definition $$P_{X,1,i}=P_{X,I,i} =  \int P_{\theta_i} d\Pi_{I,i} = \int \phi(x_i, \theta_i, \sigma^2)d\Pi_{I,i}(\theta_i) = \mathcal{N}(0,\sigma^2)*\Pi_{I,i}(x_i), \quad i\in I$$
#
# Equivalently,
#  $$P_{X,2,i}=P_{X,I^c,i} =  \int P_{\theta_i} d\Pi_{I^c,i} = \int \phi(x_i, \theta_i, \sigma^2)d\Pi_{I^c,i}(\theta_i) = \mathcal{N}(0,\sigma^2)*\Pi_{I^c,i}(x_i), \quad i \in I^c$$
#  - The **prior** is going to change for normally distributed data as no assumptions are taken on the prior (yet).
# - The **posterior** $\Pi(\mathbf{I}|X)$ is not worked seperately as it's used in $\Pi(\theta|X)$, directly.
# </details>
#
#
#
#
#
#
#
#
#


