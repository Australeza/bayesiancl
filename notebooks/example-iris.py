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
# 2. [Initialization of ProdForm](#Initialization)
# 3. [Approximation of distribution of data](#Covariance)
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

# +
#df_sample = df.sample(n=25, random_state=42)
#df_sample.to_csv("sampled25.txt", sep=";", index=False)
# -

# - Scatterplot True data clusters

sns.scatterplot(x = df["petal_area"], y= df["sepal_area"], hue=df["target"], palette="tab10")
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
bs = ProductForm(sampling_size = 500)

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

df['labels_4'] = labels_4
df['labels_3'] = labels[0]
df['labels_7'] = labels_7[0]


# - Comparison between KMeans, ProdForm and DBScan for **n_clusters = 3, 4 and 7**

# ### Plots

def flower_2d(labeling,  ax, method = "True"):
    global flower
    
    data_2 = np.array(flower)
    x_2, y_2  = data_2[:,0], data_2[:,1]
    print(len(x_2),len(y_2))
    sns.set_theme(style="white")
    sns.scatterplot(ax=ax, x=x_2, y=y_2, s=25, color=".15", hue=labeling, palette="viridis")
    ax.set_title(method)
    ax.legend().set_visible(False)
    #ax.legend(loc="upper left")
    
    return 


# # Initialization

# - ProdForm
#   
# Initialized the prior means for different "guesses" of prior means

# +
#dict_means ={"vertical_cut": [[1,30],[1,13],[1,8], [1,10]], "horizontal_cut": [[1,10],[5,10],[10,10]], "kmeans": centers_kmeans_sepal, "random": random.sample(sepal.tolist(),3)}
# -

mus = [random.sample(flower.tolist(),3), [[1,10],[5,10],[10,10]], [[1,30],[1,13],[1,8]], np.zeros((2,3)).T]
names = ["random", "initial horizontal", "initial vertical", "zeros"]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Changed from (1,4) to (2,2)
axes = axes.flatten() 
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
fig.supxlabel(f"petal area, $cm^2$")
fig.supylabel(f"sepal area, $cm^2$", x=0.05)
fig.legend(set(labels), bbox_to_anchor=(0.48,0.63))
#plt.savefig("graphs/iris_different_initial.png" , dpi=300, bbox_inches='tight')

# ## K=3

fig, axes = plt.subplots(1,3, figsize=(10, 5))
flower_2d(labels[0], axes[0], "Product- Form")
flower_2d(kmeans_labels ,axes[1],"KMeans")
flower_2d(df["target"], axes[2])
fig.suptitle("K=3")
fig.supxlabel(f"petal area, $cm^2$")
fig.supylabel(f"sepal area, $cm^2$", x=0.05)
#plt.savefig("graphs/iris_K3.png")

# ## K=4

fig, axes = plt.subplots(1,3,figsize=(10, 5) )
flower_2d(labels_4, axes[0], "Product- Form")
flower_2d(kmeans_labels_4, axes[1], "Kmeans")
flower_2d(df["target"], axes[2])
fig.suptitle("K=4")
plt.savefig("graphs/iris_K4.png")

# ## K=7

fig, axes = plt.subplots(1,3,figsize=(10, 4))
flower_2d(labels_7[0], axes[0], "Product- Form")
flower_2d(kmeans_labels_7,  axes[1], "Kmeans")
flower_2d(df["target"], axes[2])
#fig.suptitle("K=7")
fig.supxlabel(f"petal area, $cm^2$", y=-.03)
fig.supylabel(f"sepal area, $cm^2$", x=0.05)
fig.legend(set(kmeans_labels_7), bbox_to_anchor=(0.49,0.895))
#plt.savefig("graphs/iris_K7.png", dpi=300, bbox_inches='tight')

# ## Covariance

# +
from scipy.stats import pearsonr

# Dictionary to store results
pearson_results = {}

# Loop through each cluster
for label, group in df.groupby('labels_4'):
    x = group['sepal_area']
    y = group['petal_area']
    
    r, p = pearsonr(x, y)
    pearson_results[label] = {
        'Pearson r': round(r, 4),
        'p-value': round(p, 4),
        'Significant': 'Yes' if p < 0.05 else 'No'
    }

# Convert to DataFrame for inspection
import pandas as pd
pearson_df = pd.DataFrame.from_dict(pearson_results, orient='index')
print(pearson_df)

# -

# ### Normality Check: Shapiro-Wilk

# +
from scipy.stats import shapiro
from collections import Counter

results = {}
number_of_tests = 2*len(Counter(labels_7[0]).keys())
thres = 0.05/number_of_tests #adjusting for multiple testing

for label, group in df.groupby('labels_7'):
    stat1, p1 = shapiro(group['sepal_area'])
    stat2, p2 = shapiro(group['petal_area'])
    results[label] = {'W-statistic': (stat1,stat2), 'p-value': (p1,p2), 'normal1': p1 >thres, 'normal2':p2>thres}

# Convert to DataFrame for readability
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)
# -

# ## Kolmogorov-smirnov

# +
from scipy.stats import kstest, norm
import pandas as pd

# Assume df with columns 'label' and 'value'
results = {}
mus_sep = []
thres = 0.05/8
print(thres)
#bonferonni correction
for label, group in df.groupby('labels_4'):
    data = group['sepal_area']
    mu, sigma = data.mean(), data.std()
    mus_sep.append(mu)
    stat, p = kstest(data, 'norm', args=(mu, sigma))
    results[label] = {'KS-statistic': stat, 'p-value': p, 'normal': p > thres}

results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)
mus_pet = []
for label, group in df.groupby('labels_4'):
    data = group['petal_area']
    mu, sigma = data.mean(), data.std()
    mus_pet.append(mu)
    stat, p = kstest(data, 'norm', args=(mu, sigma))
    results[label] = {'KS-statistic': stat, 'p-value': p, 'normal': p > thres}

results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)

# +
import scipy.stats as stats
import matplotlib.pyplot as plt

# Sort unique labels
unique_labels = sorted(df['labels_4'].unique())

# Create a 4x2 grid (rows: 4 groups × 2 features)
fig, axes = plt.subplots(4, 2, figsize=(8, 16))  # Tall figure
fig.suptitle("QQ Plots by Cluster: Sepal and Petal Areas", fontsize=16)

for i, label in enumerate(unique_labels):
    # Sepal Area (left column)
    sepal_group = df[df['labels_4'] == label]['sepal_area']
    stats.probplot(sepal_group, dist="norm", plot=axes[i, 0])
    axes[i, 0].set_title(f"Sepal Area - Group {label}")
    axes[i, 0].set_xlabel("Theoretical Quantiles")
    axes[i, 0].set_ylabel("Sample Quantiles")
    
    # Petal Area (right column)
    petal_group = df[df['labels_4'] == label]['petal_area']
    stats.probplot(petal_group, dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(f"Petal Area - Group {label}")
    axes[i, 1].set_xlabel("Theoretical Quantiles")
    axes[i, 1].set_ylabel("Sample Quantiles")

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("graphs/qq_plots_4x2.png", dpi=300)
plt.show()

# -

# ## Posterior Means

def theta_mu(labels, prior_means, X):

    # Labels
    z = labels

    # Prior means
    posterior_means = {}

    mu_0 = prior_means

    for k in np.unique(z):
        X_k = X[z == k]
        n_k = X_k.shape[0]
        x_bar_k = X_k.mean(axis=0)
    
        # Posterior mean
        mu_post = (mu_0[k] + n_k * x_bar_k) / (1 + n_k)
    
        posterior_means[k] = mu_post

    return posterior_means


# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Extract features
X_data = df[["sepal_area", "petal_area"]].values
labels = df["labels_4"].values

# Define prior means (e.g., origin or arbitrary prior)
unique_labels = np.unique(labels)
prior_means = centers_kmeans_4

# Compute posterior means
posterior_means = theta_mu(labels, prior_means, X_data)

# Set up plot
plt.figure(figsize=(8, 6))
colors = sns.color_palette("viridis", len(unique_labels))

# Plot each group with contours using posterior means
for i, (label, group) in enumerate(df.groupby("labels_4")):
    sep = group["sepal_area"]
    pet = group["petal_area"]
    
    # Scatter the data
    plt.scatter(pet, sep, label=f"Group {label}", alpha=1, color=colors[i])
    
    # Use posterior mean instead of empirical mean
    mean = posterior_means[label]
    cov = np.cov(sep, pet)  # still empirical covariance

    # Create grid
    x = np.linspace(sep.min()-1, sep.max()+1, 100)
    y = np.linspace(pet.min()-1, pet.max()+1, 100)
    X, Y = np.meshgrid(y, x)
    pos = np.dstack((Y, X))  # petal on x-axis, sepal on y-axis
    # Compute PDF for the contour
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    # Plot contour
    plt.contour(X, Y, Z, colors=[colors[i]], alpha=0.8)

# Final touches
# Remove everything
plt.title("Clustered Iris Data", fontsize=14)
plt.xlabel("Petal area (cm$^2$)", fontsize=12)
plt.ylabel("Sepal area (cm$^2$)", fontsize=12)
plt.legend(title="Clusters", loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/iris_contours_posterior.png", dpi=300, bbox_inches='tight')
plt.show()


# +
# Assuming `flower` is a 2D numpy array and `df['labels_4']` holds cluster labels
results = []
sigma0_squared = 1  # Null hypothesis variance

for label in sorted(df['labels_4'].unique()):
    data = flower[df['labels_4'] == label]
    n = len(data)
    s_squared = np.var(data, ddof=1)  # Sample variance
    chi2_stat = (n - 1) * s_squared / sigma0_squared

    # Two-sided p-value
    p_val = 2 * min(chi2.cdf(chi2_stat, df=n-1),
                    1 - chi2.cdf(chi2_stat, df=n-1))

    results.append({
        "Cluster": label,
        "Chi-squared": round(chi2_stat, 4),
        "p-value": round(p_val, 4),
        "Significant": "Yes" if p_val < 0.05 else "No"
    })

chi2_df = pd.DataFrame(results)
print(chi2_df)


# +
#For thesis front-page- substitute this to the previous cell
# plt.axis('off')        # Turn off axes, ticks, and frame
# plt.grid(False)        # No grid
# plt.title('')          # No title
# plt.xlabel('')         # No x-label
# plt.ylabel('')         # No y-label
# plt.legend([]) 
# plt.tight_layout()
# plt.savefig("mythesis.png", dpi=300, transparent=True)
# plt.show()
# -

#




