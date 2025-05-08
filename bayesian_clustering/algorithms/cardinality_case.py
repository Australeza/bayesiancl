import time
import numpy as np
import scipy

from bayesian_clustering.algorithms.base_bayes import BaseClustering
from bayesian_clustering.card_based import marginals_i_mat, all_cmarginals, get_constant_px, \
    get_argmax_px, unique
from bayesian_clustering.card_based import mult_polynomial, get_multic, fill_cluster, \
    get_klambdas, binom_op
from bayesian_clustering.data_simulation import partition_space_nulls
from bayesian_clustering.prediction_post import partition_to1d_labels, card_post_probs, card_max_cluster, card_random_partition, repeat_sampling, \
    get_max_occur, partition_undo1d
from bayesian_clustering.priors import g_normcard, g_multicard, check_sums, normal_conv, \
    gamma_conv, anal_norm_conv_2d

np.random.seed(160)

class CardBased(BaseClustering):
    def __init__(self, cardinality_prior = None, use_partition_space: bool = False, sampling_size: int = None):
        super().__init__()
        self.data = None
        self.multi_polynomial = None
        self.polynomial_coeffs = None
        self.marginals_i = None
        self.px_const = None
        self.marginals_per_partition = None
        self.optimal_partition = None
        self.use_posterior = None
        self.g_cardinality_prior = cardinality_prior
        self.lambda_per_partition = None
        self.lambda_card = None
        self.posterior_probs = None
        self.use_partition_space = use_partition_space
        self.sampling_size = sampling_size
        if use_partition_space:
            self.sampling_size = None

    def check_input(self):
        if hasattr(self.g_cardinality_prior, 'rvs'):
            pass
        pass

    def pick_cardinality_prior(self, input_dist = None):
        if input_dist is None:
            self.g_cardinality_prior = g_normcard(self.n_points, self.n_clusters)
        else:
            self.g_cardinality_prior = input_dist

    def fit(self, data, prior_means):
        self.data = data
        self.prior_means = prior_means
        self.n_points = len(list(self.data))
        self.n_clusters = len(self.prior_means)
        print(type(data))
        #print(f"{data.ndim}d data, {self.n_clusters} clusters")
        self.marginals_i = marginals_i_mat(self.data, list(self.prior_means))
        self.multi_polynomial = mult_polynomial(self.marginals_i)
        #rint("got polynomial")
        self.polynomial_coeffs = get_multic(self.multi_polynomial,self.n_points)
        p = [0.2, 0.1, 0.7]
        print(p)
        unique_coeffs = unique(self.polynomial_coeffs)
        self.g_cardinality_prior = g_multicard(self.n_points, unique_coeffs, p)
        self.lambda_card = get_klambdas(self.g_cardinality_prior)
        #self.px_const = get_constant_px(list(self.polynomial_coeffs.values()), list(self.lambda_card.values()))
        return self

    def predict(self, data: np.ndarray):
        if self.use_partition_space:
            self._predict_partition_space()
        else:
            self._predict_using_posterior()
            self.labels.tolist()
        return self.labels

    def _predict_using_posterior(self):
        #print("Clustering using point-wise posterior...")
        #print(self.marginals_i.ndim)
        probs =  card_post_probs(self.n_points, self.marginals_i, 1, self.lambda_card)
        self.posterior_probs = probs
        if self.sampling_size is None:
            print("... and max posterior probability ...")
            self.optimal_partition = card_max_cluster(probs)
            labels = partition_to1d_labels(self.n_points,list(self.optimal_partition))
            self.labels = labels
        else:
            self.all_occurences = repeat_sampling(self.sampling_size, np.array(probs.tolist()) ,card_random_partition)
            self.labels = list(get_max_occur(self.all_occurences)[0])
            print("... and random sampling's max occurence ...")
            self.optimal_partition = partition_undo1d(self.labels, self.n_clusters)
        return self


    def _predict_partition_space(self):
        self.all_partitions = partition_space_nulls(self.n_points, self.n_clusters)
        print("Using all partition space...")
        self.marginals_per_partition, self.lambda_per_partition = all_cmarginals(self.marginals_i, list(self.all_partitions), self.lambda_card)
        optimal_idx, optimal_prod = get_argmax_px(np.array(self.marginals_per_partition),np.array(self.lambda_per_partition))
        self.optimal_partition = list(self.all_partitions)[optimal_idx]
        self.labels = partition_to1d_labels(self.n_points,list(self.optimal_partition))
        print(f"...partition space's map...{self.optimal_partition}")
        return self


    def fit_predict(self, data: np.ndarray, prior_means:np.ndarray):
        self.fit(data, prior_means)
        self.predict(data)
        self.assumptions_partition_prior_check()

    def assumptions_partition_prior_check(self):
        checks = check_sums(list(self.g_cardinality_prior.values()), self.lambda_per_partition)
        print("The assumptions on the priors are {}.".format(checks))
