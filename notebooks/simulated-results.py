# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:thesis] *
#     language: python
#     name: conda-env-thesis-py
# ---

# ## Simulations
#
# ---
# This notebook contains the results of the algorithms produced "ProductForm" and "CardBased" on the simulated datasets in "datasets" folders as well as the implementation of already existing algorithms including some visuals.
#
# **Table of content**
#
# 1. [Data simulation](#datasumulation)
# 2. [Method in steps](#section2)
# 3. [Partition estimate](#section3)
#
# ---

# +
import random
import math
import numpy as np
import ast
import multiprocessing
import time
import pandas as pd
import scipy.stats as sps
from ast import literal_eval
from more_itertools import random_combination_with_replacement

from bayesian_clustering import ProductForm
from bayesian_clustering.data_simulation import simulate_data
from bayesian_clustering.prediction_post import partition_undo1d
from bayesian_clustering.distances import evaluate_distances

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans 

# %load_ext autoreload
# %autoreload 2
# -

def flower_plot(labeling_1, labeling_2, method="True"):
    global sepal, petal
    data_1, data_2 = sepal, petal
    sns.set_theme(style="dark")
    #sepal
    x = data_1[:,0]; y=data_1[:,1]
    f, ax = plt.subplots(1,2,figsize=(10, 6))
    sns.scatterplot(ax=ax[0], x=x, y=y, s=25, color=".15", hue=labeling_1)
    #sns.histplot(ax=ax[0], x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(ax=ax[0], x=x, y=y, levels=5, color="blue", linewidths=1, alpha=0.7)
    ax[0].set_title("sepal data")
    #petal
    x_2 = data_2[:,0]; y_2=data_2[:,1]
    sns.scatterplot(ax=ax[1], x=x_2, y=y_2, s=25, color=".15", hue=labeling_2)
    #sns.histplot(ax=ax[1], x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(ax=ax[1], x=x_2, y=y_2, levels=5, color="blue", linewidths=1, alpha=0.7)
    ax[1].set_title("petal data")
    f.suptitle(method+" Labeling")
    return 


def k_means(row):
    data, k, true_label = row["generated_sample"], row["n_clusters"], row["true_labels"]
    #for 1d data otheriwise comment it out
    data = np.array(data)
    data = data.reshape(-1,1)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data)

    #labels and centroids petal
    k_means_labels = kmeans.labels_
    centers_kmeans = kmeans.cluster_centers_

    d1, d2, d3 = evaluate_distances(k_means_labels.tolist(), true_label)

    return str(k_means_labels.tolist()), str(centers_kmeans.tolist()), d1, d2, d3


def product_form(row):
    #read
    data, means, true_label = row["generated_sample"], ast.literal_eval(row["kmeans_cent"]), row["true_labels"]
    #petal
    bs = ProductForm(use_all_possible_partitions = False, use_random_sampling_posterior = False, sampling_size = 100)

    #fitting given prior means for normal data
    bs.fit(np.array(data), prior_means = means)

    #predict labels for data
    labels = bs.predict(np.array(data))[0]

    #check prior assumptions
    bs.assumptions_partition_prior_check()
    d1, d2, d3 = evaluate_distances(true_label, labels)
    return str(labels), d1, d2, d3


def do_dbscan(row):
    data, true_label = row["generated_sample"], row["true_labels"]
    #for 1d data otheriwise comment it out
    data = np.array(data).reshape(-1,1)
    db = DBSCAN(eps=0.35, min_samples=2).fit(data)
    labels = db.labels_
    d1, d2, d3 = evaluate_distances(labels.tolist(), true_label)
    return str(labels.tolist()), d1, d2, d3


deltas = [0.2, 0.5, 1.2, 3, 5.5, 9, 15]
number_of_clusters = [3, 7, 11, 15, 19, 25, 31, 35]

labels = []
dfs = {}
for k in number_of_clusters:
    min_points = math.floor(k*1.7)
    file_name = f"dataset_K{k}_N{min_points}_2000_1d.txt"
    file_path = f"../docs/datasets/simulated_datasets/{file_name}"
    df = pd.read_csv(file_path, sep="; ")

    #arrays
    df["generated_sample"] =df["generated_sample"].apply(lambda x: ast.literal_eval(x))
    df["true_labels"] =df["true_labels"].apply(lambda x:ast.literal_eval(x))
   # print(df["true_labels"])
    
    
    #kmeans
    df[["k_means_labels","kmeans_cent","kmeans_metric_rand","kmeans_metric_vi","kmeans_metric_perf"]] = \
    df.apply(lambda row: pd.Series(k_means(row)), axis=1)

    #prod-form
    df[["prodform_labels","prodform_metric_rand", "prodform_metric_vi","prodform_metric_perf"]] = \
    df.apply(lambda row: pd.Series(product_form(row)), axis=1)
    
    #dbscan
    df[["dbscan_labels","db_metric_rand", "db_metric_vi","db_metric_perf"]] = \
    df.apply(lambda row: pd.Series(do_dbscan(row)), axis=1)

    #save to dictionary with key the filename and value the produced dataframe
    #print(df.columns, df.dtypes)
    dfs[file_name] = df


# It takes more than an hour to run

newd = pd.read_csv("Results_dataset_K3_N5_2000_1d.txt'", sep = ";")
newd

newd.dtypes











