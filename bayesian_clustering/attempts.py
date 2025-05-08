import scipy.integrate as integrate
import scipy.stats as sps
import numpy as np

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
import time
import sys

def print_progress_bar(iteration, total, length=30):
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\rProgress: |{bar}| {percent:.1f}% Complete', end='')
    if iteration == total:
        print()  # Newline on completion
        


