import numpy as np
from abc import ABC, abstractmethod
from bayesian_clustering.priors import check_sums

class BaseClustering(ABC):
    """Abstract base class for product form clustering.
    """
    def __init__(self):
        self.n_clusters = None
        self.n_points = None
        self.labels = None
        self.prior_means = None

    @abstractmethod
    def fit(self, data: np.ndarray, prior_means: np.ndarray):
        pass
    @abstractmethod
    def predict(self, data: np.ndarray):
        pass
    @abstractmethod
    def fit_predict(self, data: np.ndarray, prior_means:np.ndarray):
        pass
    @abstractmethod
    def assumptions_partition_prior_check(self):
        pass

