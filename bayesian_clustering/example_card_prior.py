import numpy as np
from bayesian_clustering import CardBased

#setting seed for reproducibility
np.random.seed(160)

#data to be clustered
X = np.array([1.2,1.4,4.5,4.6,7.3,7.7])

#initialize bayesian method with cardinality based prior
bs = CardBased(use_all_possible_partitions= True, use_posterior=True, sampling_size= 200)

#fitting with prior_means
bs.fit(X, prior_means=np.array([1,2,6]))
print(bs.n_points, bs.n_clusters)

#predicting labels for data - alternatively, do: bs.fit_predict(X, np.array([2,6,8]))
labels = bs.predict(X)
print(labels)

#checking sums of priors add resp. to 1
bs.assumptions_partition_prior_check()
#print(bs.labels)

#choices for priors
card_prior_dist = {'2':('g_normcard', 'g_binomcard'),'k':'g_multicard'}
data_prior_dist = {'normal': 'normal_conv', 'gamma':'gamma_conv'}
