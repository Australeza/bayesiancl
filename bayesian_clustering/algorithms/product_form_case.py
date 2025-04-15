import numpy as np
import operator
import scipy.stats as sps

from bayesian_clustering.algorithms.base_bayes import BaseClustering
from bayesian_clustering.priors import normal_conv, sample_uniprod, check_sums, anal_norm_conv_2d
from bayesian_clustering.data_simulation import partition_space_nulls
from bayesian_clustering.prediction_post import partition_to1d_labels
from bayesian_clustering.prediction_post import prod_post_probs, prod_max_post_prob, repeat_sampling, prod_random_partition,partition_undo1d
from bayesian_clustering.product_form import compute_invconstant, omega, get_pxi, \
    normalizing_const, all_pmarginals
from bayesian_clustering.card_based import margs_mat

np.random.seed(160)

class ProductForm(BaseClustering):
    def __init__(self, use_all_possible_partitions: bool = True, use_random_sampling_posterior = False, sampling_size: int = 500 ):
        super().__init__()
        self.partition_prior_distribution = None
        self.prior_distribution = None
        self.l_values = None
        self.c_lambda =  None
        self.omegas = None
        self.marginals_i = None
        self.data = None
        self.p_const = None
        self.marginals_per_partition = None
        self.lambdas_per_partition = None
        self.optimal_partition = None
        self.use_posterior = None
        self.all_partitions = None
        self.posterior_probs = None
        self.p_sums = None
        self.use_all_possible_partitions = use_all_possible_partitions
        self.use_random_sampling_posterior = use_random_sampling_posterior
        if self.use_all_possible_partitions:
            self.use_random_sampling_posterior = False
        self.sampling_size = sampling_size

    def choose_lambda_ind(self, distribution=None, **kwargs)->np.array:
        """Assigns distribution to lambdas.

        Parameters
        ----------
        distribution: function or list or None
            distribution for individual lambdas
        kwargs
            arguments in case distribution is a function from scipy

        Returns
        -------
        np.array
            probability distribution for individual lambdas of shape k by n

        """
        if distribution is None:
            return sample_uniprod(self.n_points, self.n_clusters)
        elif hasattr(distribution, 'rvs'):
            print("has attribute rvs")
            required_arguments = set(distribution.__init__.__code__.co_varnames[1:])
            given_arguments = set(kwargs.keys())
            if not required_arguments.issubset(given_arguments) and len(given_arguments) !=0:
                raise ValueError("Incorrect arguments")
            dist = distribution
            x = np.linspace(
                dist.ppf(0.001), dist.ppf(0.999), num=self.n_clusters*self.n_points
            )
            prior_g = dist.pdf(x)
            prior_g = prior_g.reshape(self.n_clusters, self.n_points)
            sums = prior_g.sum(axis=0)
            prior_g = prior_g / sums
            return prior_g[:][:-1]
        elif isinstance(distribution,np.ndarray):
            if not self.check_distribution_properties(distribution):
                raise ValueError(r" Not a valid distribution, check for values outside (0,1) and \sum distribution_i = 1")
            return distribution
        else:
            raise ValueError("Invalid distribution")

    def check_distribution_properties(self, probabilities: np.ndarray)->bool:
        """Check if distribution is proper to be given on individual lambdas

        Parameters
        ----------
        probabilities: np.ndarray
        (n)x(k) probability matrix

        Returns
        -------
        boolean
        Checks if the probabilities are between 0 and 1, if they sum to 1 and have the correct shape
        """
        sum_check = np.sum(probabilities)==1
        domain_check = np.all(probabilities>0) and np.all(probabilities<1)
        size_check = probabilities.shape == (self.n_clusters, self.n_clusters)
        return domain_check and sum_check and size_check

    def fit(self, data, prior_means, partition_prior_distribution = None):
        self.partition_prior_distribution = partition_prior_distribution
        self.data = list(data)
        self.prior_means = prior_means
        self.n_points = len(self.data)
        self.n_clusters = len(self.prior_means)
        self.l_values = self.choose_lambda_ind(partition_prior_distribution)
        self.c_lambda = compute_invconstant(self.l_values)
        self.omegas = omega(self.l_values)
        if len(data.T) ==2:
            print('2d')
            convolution_prior = anal_norm_conv_2d
            self.marginals_i = get_pxi(self.data, self.prior_means, convolution_prior)
        else:
            self.marginals_i = get_pxi(self.data, self.prior_means)
        #print(self.marginals_i)
        self.p_sums, self.p_const = normalizing_const(self.marginals_i, self.omegas)
        return self

    def predict(self, data: np.ndarray):
        if self.use_all_possible_partitions:
            self.all_partitions = partition_space_nulls(self.n_points, self.n_clusters)
            self._predict_with_all_partitions()
        else:
            self._predict_with_posterior()
        if self.optimal_partition is None and self.labels is not None:
            #print(self.labels)
            self.optimal_partition = partition_undo1d(self.labels, self.n_clusters)
        return self.labels, self.optimal_partition


    def _predict_with_all_partitions(self):
        print("Predicting with all possible partitions...")
        self.lambdas_per_partition, self.marginals_per_partition, prod_per_partition = all_pmarginals(self.c_lambda,self.marginals_i, self.l_values, self.all_partitions)
        #print(self.c_lambda, self.p_const)
        normalized_pxs = np.array(prod_per_partition) * 1. / self.p_const
        optimal_idx = np.argmax(normalized_pxs)
        optimal_partition = list(self.all_partitions)[optimal_idx]
        #print(optimal_partition)
        self.labels = partition_to1d_labels(self.n_points,list(optimal_partition))
        print(self.labels)

    def _predict_with_posterior(self):
        print("Predicting with posterior probabilities...")
        probs = prod_post_probs(self.omegas, self.marginals_i, self.p_sums)
        self.posterior_probs = probs
        if self.use_random_sampling_posterior:
            print("Random sampling... with number of samples: {}".format(self.sampling_size))
            all_sampling_occur = repeat_sampling(self.sampling_size, np.matrix(probs),
                                                 prod_random_partition)
            max_occ_k = max(all_sampling_occur.items(), key=operator.itemgetter(1))[0]
            self.labels = list(max_occ_k)
        else:
            self.labels = prod_max_post_prob(np.matrix(probs))


    def fit_predict(self, data: np.ndarray, prior_means:np.ndarray):
        self.fit(data, prior_means)
        self.predict(data)

    def assumptions_partition_prior_check(self):
        checks = True
        if self.use_all_possible_partitions:
            checks = check_sums(self.lambdas_per_partition)
        print("The assumption on the partition prior are {}".format(checks))
