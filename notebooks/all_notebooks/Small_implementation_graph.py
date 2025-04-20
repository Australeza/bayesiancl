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
# This notebook contains the ProdForm Approach implemented for the data "C:\Users\ekaranikola1\PycharmProjects\bayesian_clustering\docs\datasets\sample_size_5_2000.txt" and the resulted graph "notebooks/graphs/k_3sample_size_incr.png"
#
#
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from bayesian_clustering import ProductForm, CardBased

# +
#import os
#print(os.getcwd())
# -

file_path = '../../docs/datasets/sample_size_5_2000.txt'
df = pd.read_csv(file_path, delimiter=',')
df = df.iloc[1:]

df.columns

df.iloc[:6]

# Convert columns to numeric if necessary
df['sample_size'] = pd.to_numeric(df['sample_size'])
df['adj_rand_index'] = pd.to_numeric(df[' adj_rand_index'])
df['var_info'] = pd.to_numeric(df[' var_info'])

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
plt.savefig('k_3sample_size_incr.png', bbox_inches='tight')
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

# +
import time
import sys

def print_progress_bar(iteration, total, length=30):
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\rProgress: |{bar}| {percent:.1f}% Complete', end='')
    if iteration == total:
        print()  # Newline on completion

# Simulate a loop
total_iterations = 100
for i in range(total_iterations + 1):
    print_progress_bar(i, total_iterations)
    time.sleep(0.05)  # simulate work

# -



