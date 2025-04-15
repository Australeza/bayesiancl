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

# ## Example Distances
# ---
# This notebook contains the ProdForm Approach implemented for the data "C:\Users\ekaranikola1\PycharmProjects\bayesian_clustering\docs\datasets\sample_size_5_2000.txt" and the resulted graph "notebooks/graphs/k_3sample_size_incr.png"
#
#
# ---

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bayesian_clustering import ProdForm

df = pd.read_csv("\docs\datasets\sample_size_5_2000.txt"

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
plt.title("Product Form, for K = 3")
plt.legend(bbox_to_anchor=(1.01, 0.93))
plt.savefig('k_3sample_size_incr.png', bbox_inches='tight')
# -








