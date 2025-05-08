# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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
import multiprocessing
import time
import pandas as pd
import scipy.stats as sps
from ast import literal_eval
from more_itertools import random_combination_with_replacement

from bayesian_clustering import ProductForm, CardBased
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


def do_kmeans(row):
    data, k, true_label = row["data"], row["clusters"], row["true_labels"]
    #for 1d data otheriwise comment it out
    if np.array(data).ndim == 1:
        data = np.array(data).reshape(-1,1)
        
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data)

    #labels and centroids petal
    kmeans_labels = kmeans.labels_.tolist()
    centers_kmeans = kmeans.cluster_centers_.tolist()
    
    #performance
    d1, d2, d3 = evaluate_distances(true_label,kmeans_labels)

    return str(kmeans_labels), str(centers_kmeans), d1, d2, d3


def do_cardbased(row):
    #read
    data, means, true_labels = row["data"], literal_eval(row["kmeans_centers"]), row["true_labels"]
    
    #petal
    bs = CardBased()

    #fitting given prior means for normal data
    bs.fit(np.array(data), prior_means = means)

    #predict labels for data
    labels = bs.predict(np.array(data))
    print(labels)

    #check prior assumptions
    #|bs.assumptions_partition_prior_check()
    d1, d2, d3 = evaluate_distances(true_labels, labels)
    return str(labels), d1, d2, d3


def do_prodform(row):
    #read
    data, means, true_label = row["data"], literal_eval(row["kmeans_centers"]), row["true_labels"]
    #petal
    bs = ProductForm()

    #fitting given prior means for normal data
    bs.fit(np.array(data), prior_means = means)

    #predict labels for data
    labels = bs.predict(np.array(data))[0][0]
    #print(labels)

    #check prior assumptions
    bs.assumptions_partition_prior_check()
    d1, d2, d3 = evaluate_distances(true_label, labels)
    return str(labels), d1, d2, d3


def do_dbscan(row):
    data, true_label = row["data"], row["true_labels"]
    #for 1d data otheriwise comment it out
    if np.array(data).ndim == 1:
        data = np.array(data).reshape(-1,1)
        
    db = DBSCAN(eps=0.35, min_samples=2).fit(data)
    labels = db.labels_.tolist()
    d1, d2, d3 = evaluate_distances(labels, true_label)
    return str(labels), d1, d2, d3


# #### Experiments

dimensions = [1, 2]
n_points = [25, 50, 75, 100, 250, 500, 750, 1000, 2000]
deltas = [0.25, 0.5, 0.75, 1, 3, 6]
n_clusters = [3, 5, 7, 11, 15]

# ## 1D

df = pd.read_csv("Dataset_1d.txt", delimiter=';') 
df.columns = df.columns.str.strip()
df.dtypes, df.shape

df[:4]

df["data"] =df["data"].apply(lambda x: literal_eval(x))
df["true_means"] =df["true_means"].apply(lambda x: literal_eval(x))
df["true_labels"] =df["true_labels"].apply(lambda x: literal_eval(x))

df[df['sample_size']==25][:5]

# ### Results 1D

# +
#kmeans
df[["kmeans_labels","kmeans_centers","kmeans_ari","kmeans_avi","kmeans_apmd"]] = \
df.apply(lambda row: pd.Series(do_kmeans(row)), axis=1)

#prod-form
df[["prodform_labels","prodform_ari", "prodform_avi","prodform_apmd"]] = \
df.apply(lambda row: pd.Series(do_prodform(row)), axis=1)

#dbscan
df[["dbscan_labels","dbscan_ari", "dbscan_avi","dbscan_apmd"]] = \
df.apply(lambda row: pd.Series(do_dbscan(row)), axis=1)
# -


#df["data"] =df["data"].apply(lambda x: x.tolist())
df.dtypes

df.columns

df.to_csv(f"Results_Dataset_1d.txt", sep=";", index=False)

df1 = pd.read_csv("Results_Dataset_1d.txt",sep=";")
df1.shape

# ### 2D

df = pd.read_csv("Dataset_2d.txt", delimiter=';') 
df.columns = df.columns.str.strip()
df.dtypes, df.shape

df["data"] =df["data"].apply(lambda x: literal_eval(x))
df["true_means"] =df["true_means"].apply(lambda x: literal_eval(x))
df["true_labels"] =df["true_labels"].apply(lambda x: literal_eval(x))

# +
#kmeans
df[["kmeans_labels","kmeans_centers","kmeans_ari","kmeans_avi","kmeans_apmd"]] = \
df.apply(lambda row: pd.Series(do_kmeans(row)), axis=1)

#prodform
df[["prodform_labels","prodform_ari", "prodform_avi","prodform_apmd"]] = \
df.apply(lambda row: pd.Series(do_prodform(row)), axis=1)

#dbscan
df[["dbscan_labels","dbscan_ari", "dbscan_avi","dbscan_apmd"]] = \
df.apply(lambda row: pd.Series(do_dbscan(row)), axis=1)
# -


df.to_csv("Results_Dataset_2d.txt", sep=";", index=False)

df



# ### CardBased
# - much slower

df = pd.read_csv("Results_Dataset_2d.txt", sep=";")
#df = pd.read_csv("Results_Dataset_2", sep=";")
df.columns

df["data"] =df["data"].apply(lambda x: literal_eval(x))
df["true_means"] =df["true_means"].apply(lambda x: literal_eval(x))
df["true_labels"] =df["true_labels"].apply(lambda x: literal_eval(x))

df = df[(df['sample_size'] == 25) & (df['clusters'] <= 3)]

#cardbased
df[["cardbased_labels","cardbased_ari","cardbased_avi","cardbased_apmd"]] = \
df.apply(lambda row: pd.Series(do_cardbased(row)), axis=1)

df['true_labels']

df.to_csv("Results_CardBased_25_3_2d.txt", sep=";", index=False)


