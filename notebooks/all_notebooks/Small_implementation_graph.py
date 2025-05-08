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

# ## Example Distances
# ---
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ast import literal_eval
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans 
from bayesian_clustering import ProductForm, CardBased
from bayesian_clustering.distances import evaluate_distances
# -

df = pd.read_csv("sampled25.txt", sep=";")

df[:4]

df["flower"] =df["flower"].apply(lambda x: literal_eval(x))
df.columns, df.dtypes

sns.scatterplot(x = df["petal_area"], y= df["sepal_area"], hue=df["target"], palette="tab10")
plt.title("True data/ clusters")

data = list(map(list,df.flower))

# +
#kmeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(data)

#labels and centroids petal
kmeans_labels = kmeans.labels_
centers_kmeans = kmeans.cluster_centers_

# +
means = centers_kmeans

#sepal
prod_prior = ProductForm(use_partition_space = False)

#fitting given prior means for normal data
prod_prior.fit(np.array(data), prior_means = means)

#predict labels for data
prod_labels = prod_prior.predict(np.array(data))[0][0]

#check prior assumptions
prod_prior.assumptions_partition_prior_check()
# -

prod_labels

# +
card_prior = CardBased()
a
#fitting given prior means for normal data
card_prior.fit(np.array(data), prior_means = means)

#predict labels for data
card_labels = card_prior.predict(np.array(data))
 
#bs.assumptions_partition_prior_check()
d1, d2, d3 = evaluate_distances(df["target"].tolist(), labels)
# -

d1

# +
fig, axes = plt.subplots(1, 3, figsize=(12, 5))  # Adjust figsize as needed

# First scatter plot
sns.scatterplot(ax=axes[0], x=df["petal_area"], y=df["sepal_area"], hue=prod_labels, palette="tab10")
axes[0].set_title("Prod Labels")

# Second scatter plot
sns.scatterplot(ax=axes[1], x=df["petal_area"], y=df["sepal_area"], hue=df["target"], palette="tab10")
axes[1].set_title("True Labels")

sns.scatterplot(ax=axes[2], x=df["petal_area"], y=df["sepal_area"], hue=card_labels, palette="tab10")
axes[2].set_title("Card Labels")

plt.tight_layout()
plt.show()
# -



# +
# Filter data: Only keep values where sample_size >= 5
df = df[df['sample_size'] >= 5]

# Set Seaborn style
sns.set(style="darkgrid")
sns.set_palette(sns.color_palette("rocket",3))

# Plot the two lines
sns.lineplot(x=df['sample_size'], y=1-df['var_info'], marker='o', label="VI score", linestyle='-', markersize=8, alpha=0.9)
sns.lineplot(x=df['sample_size'], y=df['adj_rand_index'],marker ='o',  label="ARI score", linestyle='-', markersize=8, alpha=0.9)

# Add a constant line at y = 1
plt.axhline(y=1, linestyle='-', label="Maximum metric value", alpha=0.7)

# Add labels, title, and legend
plt.xlabel("Sample Size")
plt.ylabel("Performance Metrics")
plt.ylim(bottom=0)
plt.title("Product Form, for K = 3")
plt.legend(bbox_to_anchor=(1.01, 0.93))
#plt.savefig('k_3sample_size_incr.png', bbox_inches='tight')
# -


# ###  Check input - output

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




