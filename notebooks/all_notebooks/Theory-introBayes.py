# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: kernel-spec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernel-spec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Introduction to Bayesian Analysis
# ---
#  The goal of this notebook is to introduce the basic Bayesian notions and the bayesian paradigm by using basic implementation techniques to produce graphs for intuition purposes.
#
# **Table of contents**
# 1. [Bayes Theorem](#bayestheorem)<details> <summary><i> Click here for subsection </i></summary><b> [Introductory Example](#intro-example)</b>
# </details>
# 2. [Bayesian Paradigm](#bayesian-paradigm)
# 3. [Choice of Priors](#priors) <details> <summary> <i>Click here for subsections</i>  </summary>
#   <b> [Conjugate priors](#conj-priors)</b> <br />
#      <b> [Example](#example) </b><br />
#     <b> [Non-informative priors](#non-inf-priors)</b><br />
#     <b> [2nd Example](#non-inf-example) </b>
# </details>
# 4. [Point estimation and Credible sets](#point-est-credibles)
#
# **Questions and future work**
# - tbd-1: Implement something relevant for Point estimation and Credible sets
# - tbd-2: Write formulas for a two-dim example + produce an example
#
# <details>
#   <summary><i>Details</i></summary>
#   <i>Notebook 1/2 </i>
# </details>
#
# ---

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import preliz as pz
import pymc as pm
from scipy.stats import beta

# <a id='bayestheorem'></a>
# ## __Bayes' theorem__ [[1]](https://en.wikipedia.org/wiki/Bayes%27_theorem#cite_note-17),[[2]](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2982217):
#
# $$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$
# where $A$ and $B$ are events and $P(B)\neq 0$.
#
# - $P(A|B)$ is a conditional probability: the probability of event $A$ occurring given that $B$ is true. It is also called the posterior probability of $A$ given $B$.
#
# - $P(B|A)$ is also a conditional probability: the probability of event $B$ occurring given that $A$ is true. It can also be interpreted as the likelihood of $A$ given a fixed $B$ because $P(B|A)=L(A|B)$.
#
# - $P(A)$ and $P(B)$ are the probabilities of observing $A$ and $B$ respectively without any given conditions; they are known as the prior probability and marginal probability.
#
# <a id='intro-example'></a>
# #### Introductory Example:
# #### In a certain factory machines A, B and C are all producing springs of the same length. Of their production, machines A, B and C, respectively produce 2%, 1% and 3% defective springs. Machine A produces 35% of the output of the factory, machine B 25% and machine C 40%. If one spring is selected at random from the output of the factory find the probability it is defective. If it is defective find the probability it was manufactured on machine C.
#
# Suppose $$d= \{x \text{ is defective}\}, M=\{m \; | \; m \in A,B,C\} \quad A= \{\text{x produced by A}\},\quad  \text{etc.} $$
# The problem is modeled as,
# \begin{align}
# P(d|A)=2\%, \quad P(x \in A)=35\% \\
# P(d|B) =1\%, \quad P(x \in B)=25\% \\
# P(d|C) = 3\%, \quad P(x \in C)=40 \%
# \end{align}
# Law of total probability,
# \begin{align}
# P(x\in d) = P(x \in d|A)\cdot P(A) + P(x \in d|B)\cdot P(B) + P(x \in d|C)\cdot P(C) = 2.15\%
# \end{align}
#
# Now, we need to find the probability it was manufactured from C given it is defective.
# Recall $\textbf{ Baye's Theorem}$:
# $$P(C|d) =\frac{ P(C\cap d)}{P(d)} \quad \text{ or  equivalently }\quad P(C|d) =\frac{ P(d|C) \cdot P(C)}{\sum_{m \in M} P(d|m)p(m)} \approx 55.8\%	$$
#

P_d = 0.02*0.35+0.01*0.25+0.03*0.4; print(P_d)
P_cGd = 0.03*0.4/P_d; print(P_cGd)

# <a id='bayesian-paradigm'></a>
# # The Bayesian Paradigm [[3]](https://www.routledge.com/Statistical-Theory-A-Concise-Introduction/Abramovich-Ritov/p/book/9781032007458)
#
# Suppose the observed data follow a general distribution $\mathcal{F}$ depending on $\theta$, $\quad\quad \quad x_1,x_2,.., x_n \sim\mathcal{F}(\theta)$ with density $f(x|\theta)$  and $\theta \in \mathbb{R}$ .
#
# Our belief on the data is expressed through the **prior distribution** over the parameter space, $ \quad \quad \quad \theta \sim \pi(\theta)$.
#
# *Intuitively this means that, $\theta$ is more likely to be around certain values.*
#
# *After observing the given sample and its characteristics ( pdf etc.), we update our belief to the posterior belief.*
#
# The **posterior distribution** $\pi(\theta|x)$ is defined as,
# $$\pi(\theta|x) = \frac{f(x|\theta) \, \pi(\theta) }{\int_{\Theta}f(x|\theta) \pi(\theta)\,d\theta}$$
#  where $\int_{\Theta}f(x|\theta) \pi(\theta)\,d\theta$ is the marginal distribution and $f(x|\theta)$ is the maximum likelihood. Note that  $\int_{\Theta}f(x|\theta) \pi(\theta)\,d\theta$ is independent of $\theta$ and acts as a normalizing constant.
# Equivalently, $$\pi(\theta|x) \propto f(x|\theta) \pi(\theta)$$
#
#
# $\propto$: proportional to
#

# #### The sample is given and the distribution in most cases is either given or approximated with the normal distribution. The only remaining question is _"how to pick the prior?"_
# ---
# <a id='priors'></a>
# ## Choice of Priors
# In case we hold a certain belief over the prior choice,
# <a id='conj-priors'></a>
# #### Conjugate Priors
#
# "A class of priors $\mathcal{P}$ is called conjugate for the data, if for any prior $\pi(\theta)\in P$, the corresponding posterior distribution $\pi(\theta|y)$ also belongs to $\mathcal{P}$."
#
# For example, $\theta \sim Beta(a,b)$,
#
# - Option a. Choose hyperparameters (a,b) subjectively (or through simulation).
#
#    As before,
#   \begin{align}
#  x_1,..,x_n &\sim \mathcal{F}(\theta) \\
# \theta|\alpha &\sim \pi(\alpha), \\
# \small\text{hyperparameter }&\alpha \small\text{ is known}.
# \end{align}
#
# - Option b. Put a *hyperprior* to the prior itself.
# \begin{align}
# x_1,..,x_n &\sim \mathcal{F}(\theta)\\
#  \theta|\alpha &\sim \pi(\alpha)\\
#  \alpha &\sim \psi(\alpha)
# \end{align}
#
# In case of multiple hyperpriors, we have the $\textbf{ hierarchical prior model}$.
#
#

# <a id='example'></a>
# #### Example -  *known hyperparameters*
# #### Suppose that you have a random sample $X_1, . . . , X_n \text{ from a } Ber(p)$ random variable, where p is some unknown parameter. Say we hold prior belief that $p \sim Beta(2,3)$.
#
# -In the non-bayesian context on could choose the estimate for p that minimizes the maximum likelihood; $\quad \hat{p} = \frac{\#successes}{total}$.
#
# #### Suppose that we observe on the previous context the sample with 1 and 0 corresponding to defective and non-defective:
# $$0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1$$
#
# #### then, the realized dataset  $x_1,.., x_{20} \sim Ber(p), \quad p\sim Beta(2,3)$
# $$
# \require{color}
# $$
# - The densities, $$f_{Ber}(x=1) = p, \quad \text{ and }\quad \pi(p | 2, 3) = \frac{\textcolor{blue}{{p^{2 - 1}(1 - p)^{3 - 1}}}}{B(2, 3)} = \frac{p(1 - p)^2}{B(2, 3)}, \quad 0 \leq p \leq 1$$
#
# - The Likelihood, $S =\#\text{defectives}$,
# \begin{align}
# L(x_1,..,x_{20}|p) = f_{Ber}(x_1)\cdots f_{Ber}(x_{20}) = p^{S} \cdot (1-p)^{1-S}
# \end{align}
# - The posterior,
# $$\pi(\theta|x) \propto L(x_1,..,x_{20};p) \cdot \pi(p|2,3)$$
# $$\vdots$$
# Working out the calculations we arrive at the final expression,
# $$
# \pi(p \mid x_1, \dots, x_{20}) \propto \textcolor{blue}{p^{S + 1} (1 - p)^{20 - S + 2}}$$
# Given, $S=13$, the posterior distribution is known to be:
# $$
# p \mid x_1, \dots, x_{20} \sim \text{Beta}(S + 2, 23 - S)
# $$
# #### It follows that $\pi(\theta)$ is a conjugate prior,
#  $$\pi(\theta) \sim Beta(\alpha, \beta) \quad \text{ and } \quad \pi(\theta|x) \sim Beta(\alpha ', \beta ')$$

# +
# Given data
successes = 13  # number of 1's in the sample, defective
failures = 7    # number of 0's in the sample, non-defective

# Prior parameters for Beta(2, 3)
alpha_prior = 2
beta_prior = 3

# Posterior parameters (updated with successes and failures)
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

# Estimating p by th posterior mean,
hat = ( alpha_post -1)/(alpha_post+beta_post-2)


# Define a range of probabilities from 0 to 1
p_values = np.linspace(0, 1, 500)

# Compute the density for both the prior and posterior
prior_density = beta.pdf(p_values, alpha_prior, beta_prior)
posterior_density = beta.pdf(p_values, alpha_post, beta_post)

# Plot the prior and posterior densities
plt.figure(figsize=(8, 6))
plt.plot(p_values, prior_density, label="Prior: Beta(2, 3)", color='blue', lw=2)
plt.plot(p_values, posterior_density, label="Posterior: Beta(15, 10)", color='red', lw=2)
plt.axvline(hat, label=r'MSE Estimator $\hat{p} = E(\theta|y)$', color="green")
plt.fill_between(p_values, prior_density, alpha=0.2, color='blue')
plt.fill_between(p_values, posterior_density, alpha=0.2, color='red')
plt.title('Prior and Posterior Distributions')
plt.xlabel('p')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# -

# # Intuition <a id="this-example"></a>[[example]](https://github.com/aloctavodia/BAP3/blob/main/exercises/Chapter_01.ipynb)
# In the example below, we can see that with big enough sample size, the approach converges to the true posterior density.

# +
plt.figure(figsize=(12, 9))

n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
n_heads = [ 0,  1,  2,  3,  3,  6,  8, 18, 24, 74]
theta_real = 0.5

beta_params = [(1, 1), (20, 20), (1, 4)]
x = np.linspace(0, 1, 2000)

for idx, N in enumerate(n_trials):
    if idx == 0:
        plt.subplot(4, 3, 2)
        plt.xlabel('θ')
    else:
        plt.subplot(4, 3, idx+3)
        plt.xticks([])
    y = n_heads[idx]
    for (a_prior, b_prior) in beta_params:
        posterior = pz.Beta(a_prior + y, b_prior + N - y).pdf(x)
        plt.plot(x, posterior, lw=4)

    plt.plot(theta_real, 0, ms=2, marker='o', mec='w', mfc='k')
    plt.plot(0, 0, label=f'{N:4d} trials\n{y:4d} heads', alpha=0)
    plt.xlim(0, 1)
    plt.ylim(0, 12)
    plt.legend()
    plt.yticks([])
# -

# <a id='non-inf-priors'></a>
# #### Noninformative (objective) priors
#
# In case we know nothing about $\theta$ we can pick the uniform distibution to be the prior, which is call *the non-informative prior*, all options for $\theta \in \Theta$ are equally likely,
#
# __Downsides__,
#
# 1. It can be that the parameter space has an **infinite measure**, i.e. normal distribution with $\theta = \mu, \text{ then } \Theta=(-\infty, +\infty)$.
#
#     - Distributing the probability equally from $-\infty$ to $+\infty$ would result in the so called **improper prior** (probability measure>1).
#     - Despite picking an improper prior if the posterior is proper then one can still make inference.
#
# 2. Questionable on wether to apply it to $\theta$ or its transform. As applying a transformation is no result in uniform distribution.
#
# #### Example - $1^{st}$ case.

# +
#Proper case#

# Parameters for prior (Beta(1, 1)) is the ---uniform distribution--- and posterior (Beta(7, 5))
a_prior, b_prior = 1, 1
a_post, b_post = 7, 5

# Generate values of p
p = np.linspace(0, 1, 500)

# Calculate the prior and posterior densities
prior_pdf = beta.pdf(p, a_prior, b_prior)
posterior_pdf = beta.pdf(p, a_post, b_post)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(p, prior_pdf, label='Prior Beta(1, 1)', color='blue', lw=2)
plt.plot(p, posterior_pdf, label='Posterior Beta(7, 5)', color='red', lw=2)
plt.fill_between(p, 0, prior_pdf, color='blue', alpha=0.2)
plt.fill_between(p, 0, posterior_pdf, color='red', alpha=0.2)
plt.title('Prior and Posterior Distributions', fontsize=14)
plt.xlabel('p', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# -

# <a id='non-inf-example'></a>
# ### Example - $2^{nd}$ case
#
# #### Suppose $x_1, x_2, \dots, x_{10} \sim \mathcal{N}(\mu, \sigma^2)$, where $ \mu \in (-\infty, +\infty) \text{ and } \sigma^2 = 0.04$ is known, and now, assume an improper prior $ \pi(\mu) \propto 1$.
#
# * The non-informative prior reflects no prior information on the parameter by assuming uniformity, $\pi(\mu) = c$, with $c\in [0,1] $,
# * The probability measure should integrate to 1 but $\; \int_{-\infty}^{+\infty} c = +\infty$ which makes it improper.
#
# **Applying the Bayesian paradigm**
# - The **likelihood**  $\mathcal{L}(\mu) \propto \exp\left(-\frac{n}{2\sigma^2} (\mu - \bar{x})^2\right)$
#   where $\bar{x}$ is the sample mean.
#
# * Since the prior is improper (fixed) , the posterior is proportional to the likelihood:
#
# $$
# \pi(\mu \mid x_1, \dots, x_{10}) \propto \exp\left(-\frac{n}{2\sigma^2} (\mu - \bar{x})^2\right)\newline$$
#
# * Thus, the **posterior distribution** is:
# $$
# \mu \mid x_1, \dots, x_{10} \sim \mathcal{N}\left( \bar{x}, \frac{\sigma^2}{n} \right)
# $$
#
# #### _Remark_, we started with an improper prior and arrived at a well known condition for $\mu$.
# Conclusion: The produced plots depict the posterior densities for small(left) and big(right) n. It is obvious that the densities are concentrated around the true mean.
#

# +
# Construction of datasets
n_small = 5
n_big = 40
true_mean = 4 #true mean
data5 = pz.Normal(mu=true_mean, sigma=4).rvs(n_small) #synthetic sample of size n_small = 5
data40 = pz.Normal(mu=true_mean, sigma=4).rvs(n_big) #sythetic sample of size n_big = 40

# Values to be approximated by the posterior dist
print(np.mean(data5),np.mean(data40))

# Bayesian analysis for small and big sample
idatas = []
for data in [data5,data40]:
    with pm.Model() as model:
        m = pm.Normal('m', mu=1, sigma=1) #uniform prior
        y = pm.Normal('y', mu= m, observed=data) #N(m, sigma)
        idata = pm.sample(random_seed=123)
        idatas.append(idata)

#plotting the posterior density of mu
_, axes = plt.subplots(1,2, figsize=(12, 4))
for idata, ax in zip(idatas, axes):
    az.plot_posterior(idata, ax=ax)
    
#print(idatas[0])
# -

# <a id='point-est-credibles'></a>
# ## Point Estimation and Credible Sets
# #### Measures of goodness-of-estimation
# * Mean Squared Error: To estimate $\theta$ by $\hat{\theta}$, one can minimize the MSE
# $$MSE(\theta) = \mathbb{E}_{\theta} (\theta-\hat{\theta}|y)^2$$ which is minimized when  $\hat{\theta} =\mathbb{E}_{\theta}(\theta|y)$
#
# * Mean Absolute Error: The estimator $\theta$ with respect to the mean absolute error is the posterior
# median $\hat{\theta}= Med(\theta|y)$.
# #### Credible Sets: analog to Confidence Interval
#
# _A $100(1−\alpha)\%$ confidence set $S_y \subset \Theta$ is defined by: $$P(\theta \in S_y|y) = 1−\alpha$$_

#












