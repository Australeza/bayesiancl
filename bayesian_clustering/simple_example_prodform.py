import random

import numpy as np
import multiprocessing
import time

import scipy.stats as sps
from more_itertools import random_combination_with_replacement

from bayesian_clustering import ProductForm
from bayesian_clustering.data_simulation import simulate_data
from bayesian_clustering.prediction_post import partition_undo1d
from bayesian_clustering.distances import evaluate_distances

##https://stackoverflow.com/questions/492519/timeout-on-a-function-call

#Setting seed for reproducibility
np.random.seed(42)

#number of observations
n = 7

#number of clusters
k = 3

#example data array matching n=4
X = np.array([10,11,1.2,1.4,4.5,4.6,7,8,9])

#prior on individual points

#1. example distribution for obs-cluster probability
v = np.array([1,2,2,5,5,7,7,1.5,1.7, 3.2,5.5,6])
example_oc_probs = v/np.sum(v)

#2. example distribution from sps
dists = {'probability vector': v, 'normal':sps.norm}

#prior on data
#initializing bayesian method for product-form
bs = ProductForm(use_all_possible_partitions = False, use_random_sampling_posterior = True, sampling_size = 300)

#fitting given prior means for normal data
bs.fit(X, prior_means = np.array([8,3,1]), partition_prior_distribution = dists['normal'])

#predict labels for data
labels = bs.predict(X); print(labels, X)

#check prior assumptions
bs.assumptions_partition_prior_check()

#choices for prior on partition
input_dist_ind_probs = {'2': 'sample_uniprod'}
