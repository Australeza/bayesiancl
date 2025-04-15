import scipy.integrate as integrate
import itertools
import scipy.stats as sps
import numpy as np



def check_sums(g_m:list, lambdas_i:list = None)-> bool|tuple:
    """Checking assumptions

    Parameters
    ----------
    g_m: list
        prior values
    lambdas_i: list
        weights for all clusterings

    Returns
    -------
    boolean
        True if sum of values for each list of floats adds up to 1.
    """
    sum_card = round(np.sum(list(g_m)),3)
    if sum_card == 1.0:
        if lambdas_i is None:
            return True
        elif round(np.sum(list(lambdas_i)),3) == 1.0:
            return True
    return False

#theta-priors 1-d
def normal_conv(x_i: float, mu:float) -> float:
    r"""
    The convolution between the distribution of the data (normal) and prior distribution (normal).

    Parameters
    ----------
    x_i: float
        data point
    mu: float
        prior mean
    Returns
    -------
    float
        Convoluted value \int f_{N(\theta, 1)}(x_i) g_{N(mu,1)}(\theta) d\theta.
    """
    value = integrate.quad(lambda theta: sps.norm.pdf(x_i, theta, 1)*sps.norm.pdf(theta, mu, 1),-np.inf,+np.inf)
    return value[0]

#theta-priors 2-d
#theta-priors
def normal_conv_2d(x_i: np.array, prior_mu:np.array) -> float:
    r"""
    The convolution between the distribution of the data (normal) and prior distribution (normal).

    Parameters
    ----------
    x_i: float
        data point
    prior_mu: float
        prior mean
    Returns
    -------
    float
        Convoluted value \int f_{N(\theta, 1)}(x_i) g_{N(mu,1)}(\theta) d\theta.
    """
    f = lambda theta_1, theta_2:  sps.multivariate_normal.pdf(x_i, mean= (theta_1, theta_2), cov= np.eye(2))* sps.multivariate_normal.pdf((theta_1, theta_2), mean = prior_mu, cov= np.eye(2))
    value = integrate.dblquad(f, -np.inf, +np.inf, -np.inf, +np.inf)
    return value[0]

def anal_norm_conv_2d(x_i: np.array, prior_mu:np.array) -> float:
    """
    This function computes the derived analytical formula in order to potentially reduce the complexity of the 2-dimensional convolution from normal_conv_2d.
    Parameters
    ----------
    x_i: np.array
        data point

    prior_mu: np.array
        prior means

    Returns
    -------
    float
        The value of the derived expression at point x_i with prior mean(2d) prior_mu.
    """
    value = (1./(4*np.pi))*np.exp(-0.25*((x_i[0]-prior_mu[0])**2+(x_i[1]-prior_mu[1])**2))
    return value

def gamma_conv(x_i: float, a:float) -> float:
    r"""
    The convolution between the distribution of the data (normal) and prior distribution (gamma).

    Parameters
    ----------
    x_i: float
        data point
    a: float
        hyper-parameter of the gamma distribution

    Returns
    -------
    float
        Convoluted value \int f_{N(\theta, 1)}(x_i) g_{Gamma(a,1)}(\theta) d\theta, \theta \in (0,+oo).
    """
    value = integrate.quad(lambda theta: sps.norm.pdf(x_i, theta, 1)* sps.gamma.pdf(theta, a),0,+np.inf)
    return value[0]

#cardinality priors
def g_normcard(m_size:int,  k:int, sigma:int =1,)-> list:
    """
    Maps values of the normal distribution to all cardinalities.

    Parameters
    ----------
    m_size: int
        size of all cardinalities
    k: int
        number of clusters
    sigma: float = 1
        standard deviation

    Returns
    -------
    list
        values for each indexed cardinality for both K=2 and K>2
    """
    dist = sps.norm(loc=0, scale=sigma)
    if k == 2:
        m_size = m_size-1
    x = np.linspace(
        dist.ppf(0.001), dist.ppf(0.999), num=m_size
    )  # chooses n points between the 0.001 quantile and the 0.999 quantile.
    prior_g = dist.pdf(x) / dist.pdf(x).sum()
    return list(prior_g)

#K=2
def g_binomcard(n:int, m:list[int], p:float) -> list:
    """
    Maps values of the binomial distribution to all cardinalities (biclustering case).

    Parameters
    ----------
    n: int
        number of cardinalities
    m: list
        list of cardinalities
    p: float
        probability of success of the binomial distribution

    Returns
    -------
    list
        Values for all cardinalities from the binomial distribution.
    """
    dist = sps.binom(n,p)
    values = [dist.pmf(k) for k in m]
    values = np.array(values)/sum(values)
    return list(values)

#K>2
def g_multicard(n:int, m:list[tuple], p:list[float]) -> dict:
    """
    Maps values of the multi-card distribution to all cardinalities (clustering case).

    Parameters
    ----------
    n: int
    number of data points
    m: list
        list of cardinalities
    p: list
        probabilities of success for each cluster from the multinomial distribution

    Returns
    -------
    list
        Values for all cardinalities from the multinomial distribution.
    """
    dist = sps.multinomial(n,p)
    v_uniq = {k: dist.pmf(list(k)) for k in m}
    v_uniq = {k: v/np.sum(list(v_uniq.values())) for k, v in v_uniq.items()}
    v = {}
    for k,value in v_uniq.items():
        permuted = permute_combs([k])
        new_value = value/len(tuple(permuted))
        v.update({p:new_value for p in permuted})
    values = {vs: v[vs] for vs in v.keys()}
    return values


## Product-case
def sample_uniprod(n: int,k: int)-> np.array:
    """
    Produces small lambdas from the uniform distribution.
    Parameters
    ----------
    n: int
        number of data points
    k: int
        number of clusters

    Returns
    -------
    np.array
        Sample of small lambdas for each combination (cluster, point) of dimension (k-1)x(n)
    """

    values = np.random.rand(k, n)
    norms = np.sum(values, axis=0)
    values = values / norms
    return values[:][:-1]

def permute_combs(given_p:list)->set:
    """Permutes cardinalities.

    Parameters
    ----------
    given_p: list
        list of cardinalities

    Returns
    -------
    set
        New set of cardinalities with permutations.

    """
    res_part = []
    for u_part in given_p:
        perm = itertools.permutations(u_part)
        #perm = map(tuple, perm)
        res_part = res_part + list(perm)
    return set(res_part)
