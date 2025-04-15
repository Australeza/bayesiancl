import random

import numpy as np
import multiprocessing
import time
import math

import scipy.stats as sps
from more_itertools import random_combination_with_replacement

from bayesian_clustering import ProductForm
from bayesian_clustering.data_simulation import simulate_data
from bayesian_clustering.prediction_post import partition_undo1d
from bayesian_clustering.distances import evaluate_distances

def save_to_file(sample, predicted_labels, true_label, d, timing, delta, k):
    with open(f"../docs/datasets/sample_size_5_2000.txt", "a") as file:
        sample_to_str = " ".join(map(str, sample))
        predicted_label_to_str = " ".join(map(str, predicted_labels))
        true_label_to_str = " ".join(map(str, true_label))
        add_tofile = f"{0}; {delta}; {k}; {len(sample)}; {sample_to_str}; {predicted_label_to_str}; {true_label_to_str}; {d[0]}; {d[1]}; {d[2]}; {timing} "
        file.write(add_tofile + "\n")
    return

def varying_sample_size(sample, true_label, delta, K):
    #prior options
    v = np.zeros(shape=(3, 3))
    dists = {'normal': sps.norm}

    #time it
    t_o = time.time()

    #model
    model = ProductForm(use_all_possible_partitions=False,
                    use_random_sampling_posterior=False, sampling_size=100)
    # fitting given prior means for normal data and partition prior normal
    model.fit(sample, prior_means=np.array([2, 6, 11]),
                   partition_prior_distribution=dists['normal'])
    # predict labels for data
    model.predict(sample)
    predicted_labels = list(model.labels)

    #distances
    d1, d2, d3 = evaluate_distances(predicted_labels, true_label)

    #time it
    duration_of_run = time.time() - t_o
    print("time in seconds:", duration_of_run)

    #write on file
    save_to_file(sample, predicted_labels, true_label, [d1, d2, d3], duration_of_run, delta, K)

    print("performance:", len(sample), d1, d2, d3)

    return predicted_labels

def means_delta_distance(n_clusters:int, delta:float)-> np.array:
    """Created the prior means based on a distance delta

    Parameters
    ----------
    n_clusters: int
        number of clusters
    delta: float
        step for 1d prior means

    Returns
    -------
    np.array
        An array of prior means centered at 1 in the following way:
        i.e. delta = 0.2 and K = 3 then [1-0.2, 1, 1+0.2]
            delta = 0.4 and K = 5 then [1-0.4*2, 1-0.4*1, 1, 1+0.4*1, 1+0.4*2]
            etc.
    """
    one_side = math.modf(n_clusters/2)[1]
    produced_means = np.linspace(1-one_side*delta, 1+ one_side*delta, num=n_clusters)
    return produced_means

def main():
    #maximum duration, after that interrupt process
    max_duration = 1800  # 30 minutes

    # Varying sample size N
    highest_n  = 2000
    sample_sizes = np.linspace(5, highest_n, 10, dtype=int)

    # Fixed number of clusters
    K = 3

    # distances between true means
    deltas = [0.2, 0.5, 1.2, 3, 5.5, 9, 15]

    ## Data Simulation ##
    for delta in deltas:

        #true means with distance delta
        true_means = means_delta_distance(n_clusters=K, delta=delta)
        #Initialize labels list, partitions list
        true_labels, true_partitions, simulated_samples = [], [], []

        # Sample generation of different sample sizes
        for sample_size in sample_sizes:
            label = list(random_combination_with_replacement(range(K), sample_size))
            random.shuffle(label)
            partition = list(partition_undo1d(list(label), K))
            true_labels.append(list(label))
            true_partitions.append(partition)
            mixture, sampled_x = simulate_data(true_means, partition, sample_size)
            simulated_samples.append(sampled_x.tolist())
            if sample_size == 4 or sample_size < 160:
                print(sampled_x.tolist(), label)

        with open("../docs/datasets/sample_size_5_2000.txt", "w") as file:
            file.write("type, delta, n_clusters, sample_size, generated_sample, predicted_labels, true_labels, adj_rand_index, var_info, perfect_matching, time \n")
            for idx, sample in enumerate(simulated_samples):
                print(idx)
                #process
                process = multiprocessing.Process(target=varying_sample_size, args=(sample,true_labels[idx], delta, K))
                process.start()
                process.join(max_duration+0.01)

            if process.is_alive():
                process.terminate()
                process.join()
                print("took longer than {} seconds".format(max_duration))



if __name__ == "__main__":
    main()
