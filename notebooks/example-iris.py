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

# ## Dataset Iris
#
# ---
# This notebook contains the results of the algorithms produced "ProductForm" on the iris dataset provided by sklearn as well as the implementation of already existing algorithms including some visuals.
#
# **Table of content**
#
# 1. [Data Aggregation](#data)
# 2. [Comparison Between Methods](#Comparison)
# 3. [Initialization of ProdForm](#Initialization)
#
# ---

# +
import random
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sps
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN

from bayesian_clustering import ProductForm, CardBased

# %load_ext autoreload
# %autoreload 2
# -

# # Data

# - Loading iris from sklean

iris = load_iris()
#iris.target

df = pd.DataFrame(iris.data, columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"])
df['target'] = iris.target
print(df[:5])

# ### - Aggregating the columns into one *2-d* column
#   Rough approximation of the "area" covered by the petal and the sepal of each flower
#   
#   i.e. sepal_area = ["sepal length" x "sepal width"]

# +
df["sepal_area"] = df.apply(lambda row: row['sepal_len']*row['sepal_wid'],axis =1)
df["petal_area"] =  df.apply(lambda row: row['petal_len']*row['petal_wid'],axis =1)

df["flower"] = df.apply(lambda row: [row['petal_area'], row['sepal_area']], axis =1)
print(df[:4])
# -

# flower = [sepal_area, petal_area]

flower = np.array(list(map(list,df.flower)))
flower[:4].tolist()

# - Scatterplot True data clusters

sns.scatterplot(x = df["petal_area"], y= df["sepal_area"], hue=df["target"])
plt.title("True data/ clusters")

# # Comparison

# ### **n_clusters = 3**

# - KMeans

# +
#kmeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(flower)

#labels and centroids petal
kmeans_labels = kmeans.labels_
centers_kmeans = kmeans.cluster_centers_
# -

# - ProdForm

# +
means = centers_kmeans

#sepal
bs = ProductForm(use_all_possible_partitions = False, use_random_sampling_posterior =False, sampling_size = 500)

#fitting given prior means for normal data
bs.fit(np.array(flower), prior_means = means)

#predict labels for data
labels = bs.predict(np.array(flower))[0]

#check prior assumptions
bs.assumptions_partition_prior_check()
# -

# ### **n_clusters = 4**

# - KMeans

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(flower)
#labels and centroids petal
kmeans_labels_4 = kmeans.labels_
centers_kmeans_4 = kmeans.cluster_centers_

# - ProdForm

# +
means = centers_kmeans_4

#sepal
bs = ProductForm(use_all_possible_partitions = False, use_random_sampling_posterior =True, sampling_size = 500)

#fitting given prior means for normal data
bs.fit(np.array(flower), prior_means = means)

#predict labels for data
labels_4 = bs.predict(np.array(flower))[0]

#check prior assumptions
bs.assumptions_partition_prior_check()
# -

# ### **n_clusters = 7**

# - KMeans

#kmeans
kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(flower)
#labels and centroids petal
kmeans_labels_7 = kmeans.labels_
centers_kmeans_7 = kmeans.cluster_centers_

# - ProdForm

# +
mean = centers_kmeans_7
#sepal
bs = ProductForm(use_all_possible_partitions = False, use_random_sampling_posterior =False, sampling_size = 500)

#fitting given prior means for normal data
bs.fit(flower, prior_means = means)

#predict labels for data
labels_7 = bs.predict(np.array(flower))[0]

#check prior assumptions
bs.assumptions_partition_prior_check()


# -


print(labels)


# - Comparison between KMeans, ProdForm and DBScan for **n_clusters = 3, 4 and 7**

# ### Plots

def flower_2d(labeling,  ax, method = "True"):
    global flower
    
    data_2 = np.array(flower)
    x_2, y_2  = data_2[:,0], data_2[:,1]
    print(len(x_2),len(y_2))
    
    sns.scatterplot(ax=ax, x=x_2, y=y_2, s=25, color=".15", hue=labeling)
    ax.set_title(method)
    
    return 


fig, axes = plt.subplots(1,3, figsize=(10, 5))
flower_2d(labels[0], axes[0], "Product- Form")
flower_2d(kmeans_labels ,axes[1],"KMeans")
flower_2d(df["target"], axes[2])
fig.suptitle("K=3")
plt.savefig("graphs/iris_K3.png")

fig, axes = plt.subplots(1,3,figsize=(10, 5) )
flower_2d(labels_4, axes[0], "Product- Form")
flower_2d(kmeans_labels_4, axes[1], "Kmeans")
flower_2d(df["target"], axes[2])
fig.suptitle("K=4")
plt.savefig("graphs/iris_K4.png")

fig, axes = plt.subplots(1,3,figsize=(10, 5) )
flower_2d(labels_7[0], axes[0], "Product- Form")
flower_2d(kmeans_labels_7,  axes[1], "Kmeans")
flower_2d(df["target"], axes[2])
fig.suptitle("K=7")
plt.savefig("graphs/iris_K7.png")

# # Initialization

# - ProdForm
#   
# Initialized the prior means for different "guesses" of prior means

# +
#dict_means ={"vertical_cut": [[1,30],[1,13],[1,8], [1,10]], "horizontal_cut": [[1,10],[5,10],[10,10]], "kmeans": centers_kmeans_sepal, "random": random.sample(sepal.tolist(),3)}
# -

mus = [random.sample(flower.tolist(),3), [[1,10],[5,10],[10,10]], [[1,30],[1,13],[1,8]], np.zeros((2,3)).T]
names = ["random", "initial horizontal", "initial vertical", "zeros"]

fig, axes= plt.subplots(1,4, figsize=(13,5))
for idx, means in enumerate(mus):
    #flower
    bs = ProductForm(use_all_possible_partitions = False, use_random_sampling_posterior =True, sampling_size = 500)

    #fitting given prior means for normal data
    bs.fit(np.array(flower), prior_means = means)

    #predict labels for data
    labels = bs.predict(np.array(flower))[0]

    #check prior assumptions
    bs.assumptions_partition_prior_check()
    flower_2d(labels, axes[idx], names[idx])
plt.savefig("graphs/iris_different_initial.png")


















