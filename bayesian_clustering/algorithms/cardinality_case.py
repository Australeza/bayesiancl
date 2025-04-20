import time
import numpy as np
import scipy

from bayesian_clustering.algorithms.base_bayes import BaseClustering
from bayesian_clustering.card_based import margs_mat, get_1d_polynomial, get_coeffs, \
    get_subsets_m, all_cmarginals, get_constant_px, \
    get_argmax_px, get_lambdas, fill_coeffs
from bayesian_clustering.card_based import mult_polynomial, get_multic, fill_cluster, \
    get_klambdas, binom_op, unique
from bayesian_clustering.data_simulation import partition_space_nulls
from bayesian_clustering.prediction_post import partition_to1d_labels, card_post_probs, \
    card_comp_post, card_max_cluster, card_random_partition, repeat_sampling, \
    get_max_occur, partition_undo1d, \
    card_k_comp_post
from bayesian_clustering.priors import g_normcard, g_multicard, check_sums, normal_conv, \
    gamma_conv, anal_norm_conv_2d

np.random.seed(160)

class CardBased(BaseClustering):
    def __init__(self, cardinality_prior = None, use_all_possible_partitions: bool = False, use_posterior: bool = True, sampling_size: int = None):
        super().__init__()
        self.multi_polynomial = None
        self.g_prior_dictionary = None
        self.multi_coeffs_dict = None
        self.mult_polynomial_coeffs = None
        self.mult_polynomial = None
        self.polynomial_coeffs = None
        self.polynomial_1d = None
        self.marginals_i = None
        self.data = None
        self.px_const = None
        self.marginals_per_partition = None
        self.optimal_partition = None
        self.use_posterior = None
        self.g_per_partition = None
        self.all_partitions = None
        self.posterior_probs = None
        self.g_cardinality_prior = cardinality_prior
        self.lambda_per_partition = None
        self.lambda_per_cardinality = None
        self.posterior_probs = None
        self.use_all_possible_partitions = use_all_possible_partitions
        self.use_posterior = use_posterior
        if use_all_possible_partitions == use_posterior:
            self.use_all_possible_partitions = False
            self.use_posterior = True
        self.sampling_size = sampling_size

    def check_input(self):
        if hasattr(self.g_cardinality_prior, 'rvs'):
            pass
        pass

    def get_prior_per_m(self) -> dict:
        all_ms = self.mult_polynomial_coeffs
        prior_values = self.g_cardinality_prior
        prior_dict = dict.fromkeys(all_ms)
        for idx, m in enumerate(all_ms):
            prior_dict[m] = prior_values[m]
        return prior_dict
    def check_assumptions(self):
        return check_sums(self.g_cardinality_prior)
    def get_lambda_card_posterior(self):
        g_m = self.g_cardinality_prior
        return [scipy.special.binom(self.n_points, i + 1) ** (-1) * g_m[i] for i in
                        range(len(g_m))]

    def get_dict_lambdas_posterior(self):
        lambdas_m = {m: self.g_cardinality_prior[m] * 1. / binom_op(m) for m in self.g_cardinality_prior.keys()}
        return lambdas_m

    def pick_cardinality_prior(self, input_dist = None):
        if input_dist is None:
            self.g_cardinality_prior = g_normcard(self.n_points, self.n_clusters)
        else:
            self.g_cardinality_prior = input_dist

    def fit(self, data, prior_means, data_prior = normal_conv):
        self.data = list(data)
        self.prior_means = prior_means
        self.n_points = len(list(self.data))
        self.n_clusters = len(self.prior_means)
        print(type(data))
        if len(data.T) == 2:
            print("2d")
            self.marginals_i = margs_mat(list(self.data), list(self.prior_means),
                                         anal_norm_conv_2d)
        else:
            self.marginals_i = margs_mat(list(self.data), list(self.prior_means), normal_conv)

        if self.n_clusters == 2:
            if self.g_cardinality_prior is None:
                self.pick_cardinality_prior()
            self.polynomial_1d = get_1d_polynomial(self.marginals_i)
            self.polynomial_coeffs = get_coeffs(self.polynomial_1d)
        else:
            print(">3 clusters")
            self.multi_polynomial = mult_polynomial(self.marginals_i)
            print("got coeffs")
            mult_poly_coeffs = get_multic(self.multi_polynomial)
            self.mult_polynomial_coeffs = fill_coeffs(self.n_points, mult_poly_coeffs)
        return self

    def predict(self, data: np.ndarray):
        if self.n_clusters == 2:
            self._predict_cardinality_k2()
        else:
            self._predict_general_cardinality()
        return self.labels, self.optimal_partition


    def _predict_cardinality_k2(self):
        if self.use_all_possible_partitions:
            print("Two clusters using all possible partitions...")
            self._predict_using_all_partition_for2()
            self.lambda_per_partition, self.lambda_per_cardinality = get_lambdas(
                self.g_cardinality_prior, self.all_partitions, self.n_points)
            self.px_const = get_constant_px(list(self.polynomial_coeffs),
                                            self.lambda_per_cardinality)
            self.marginals_per_partition = np.array(
                self.marginals_per_partition) * self.px_const ** (-1)
            optimal_idx, optimal_prod = get_argmax_px(self.marginals_per_partition,
                                                      np.array(
                                                          self.lambda_per_partition))
            first_cluster = self.all_partitions[optimal_idx]
            self.optimal_partition = fill_cluster(self.n_points, first_cluster)
            self.labels = partition_to1d_labels(list(self.optimal_partition))
        else:
            print("Two clusters using posterior probabilities...")
            self.lambda_per_cardinality = self.get_lambda_card_posterior()
            self.px_const = get_constant_px(list(self.polynomial_coeffs),
                                            self.lambda_per_cardinality)
            self._predict_using_posterior()
    def _predict_using_all_partition_for2(self):
        part_space = []
        resulted_margs = []
        start_time = time.time()
        for m in range(1, self.n_points):  # 1, 2, 3, 4, 5
            subsets_m = get_subsets_m(self.n_points, m)
            marginal_per_card = all_cmarginals(self.marginals_i, subsets_m)
            part_space.extend(subsets_m)
            resulted_margs.extend(marginal_per_card)
            print("--- %s seconds ---" % (time.time() - start_time))
        self.all_partitions = part_space
        self.marginals_per_partition = resulted_margs
        #print(type(self.all_partitions), type(self.g_cardinality_prior, ))

    def _predict_using_posterior(self):
        if self.n_clusters == 2:
            probs = card_post_probs(self.n_points, self.marginals_i, 1, self.lambda_per_cardinality, card_comp_post)
        else:
            probs =  card_post_probs(self.n_points, self.marginals_i, 1, self.lambda_per_cardinality, card_k_comp_post)
        self.posterior_probs = probs
        if self.sampling_size is None:
            print("... and max posterior probability ...")
            self.optimal_partition = card_max_cluster(probs.tolist())
            self.labels = partition_to1d_labels(self.n_points,list(self.optimal_partition))
        else:
            self.all_occurences = repeat_sampling(self.sampling_size, np.array(probs.tolist()) ,card_random_partition)
            self.labels = list(get_max_occur(self.all_occurences))
            print("... and random sampling's max occurence ...")
            self.optimal_partition = partition_undo1d(self.labels, self.n_clusters)



    def _predict_general_cardinality(self):
        unique_cardinalities = unique(self.mult_polynomial_coeffs).keys()
        self.g_cardinality_prior = g_multicard(self.n_points, unique_cardinalities,
                                               [0.2, 0.1, 0.7])
        if self.use_all_possible_partitions:
            self.all_partitions = partition_space_nulls(self.n_points, self.n_clusters)
            print(self.mult_polynomial_coeffs)
            print("Using all possible partitions...for {} groups".format(self.n_clusters))
            self.marginals_per_partition = all_cmarginals(self.marginals_i, list(self.all_partitions))
            self.lambda_per_partition,self.lambda_per_cardinality = get_klambdas(dict(self.g_cardinality_prior),list(self.all_partitions))
            #self.px_const = get_constant_px(list(self.mult_polynomial_coeffs.values()), list(self.lambda_per_cardinality.values()))
            optimal_idx, optimal_prod = get_argmax_px(np.array(self.marginals_per_partition),np.array(self.lambda_per_partition))
            self.optimal_partition = list(self.all_partitions)[optimal_idx]
            self.labels = partition_to1d_labels(list(self.optimal_partition))
        else:
            print("Clustering using max posterior density...")
            #self.g_prior_dictionary = self.get_prior_per_m()
            self.lambda_per_cardinality = self.get_dict_lambdas_posterior()
            #self.px_const = get_constant_px(list(self.mult_polynomial_coeffs.values()),list(self.lambda_per_cardinality.values()))
            self._predict_using_posterior()

    def fit_predict(self, data: np.ndarray, prior_means:np.ndarray):
        self.fit(data, prior_means)
        self.predict(data)

    def assumptions_partition_prior_check(self):
        checks = None
        if self.n_clusters == 2:
            checks = check_sums(self.g_cardinality_prior,self.lambda_per_cardinality)
        elif isinstance(self.g_cardinality_prior, dict):
            checks = check_sums(list(self.g_cardinality_prior.values()), self.lambda_per_partition)
        elif isinstance(self.g_cardinality_prior, list):
            checks = check_sums(list(self.g_cardinality_prior), self.lambda_per_partition)
        print("The assumption on the cardinality - prior and partition-prior are (sum of resp values is 1) {}".format(checks))
