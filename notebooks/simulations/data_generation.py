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

# # Clustering in K groups - simulation
#
# ---
# This notebook contains the code that produced the 1D and 2D datasets.
#
# **Table of content**
#
# ---

# +
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

from bayesian_clustering.data_simulation import initialize_true, simulate_data, simulate_2d_data


# -

def create_1d_means(delta:float, k:int)->np.array:
    # Create an empty vector of size k
    vector = np.zeros(k)
    
    # Find the middle index
    mid = k // 2
    
    # Set the middle element to 1
    vector[mid] = 1
    
    # Fill the vector by stepping with delta
    for i in range(1, mid + 1):
        vector[mid - i] = 1 - i * delta
        vector[mid + i] = 1 + i * delta
    
    return vector.tolist()


def create_2d_means(delta, k):
    array = np.zeros((k, 2))
    
    # Middle of the array
    mid_x = k // 2
    array[mid_x] = [1,1]
    
    for i in range(k):
        # Index distance to middle point
        if i == mid_x:
            continue
        dist = abs(i-mid_x)
        r = delta*dist
        y = random.uniform(1-r,1+r)
        rand_sign = random.choice([-1, 1])
        x = 1 + rand_sign*math.sqrt(r**2-(1-y)**2)
        array[i] = [x, y]  
    return array.tolist()


# ### Generation function
# - run only if you wish to overwrite the given datasets

def simulate_datasets(deltas, n_clusters, n_points, dim = 1):
    #File
    file_name = f'Dataset_{dim}d.txt'
    with open(file_name, "w") as file:
        file.write(f'dimensions;clusters;delta;sample_size;data;true_means;true_labels \n')
        for k in n_clusters:
            for n in n_points:
                for delta in deltas:
                    true_labels, true_partition = initialize_true(n, k)
                    if dim == 2:
                        means = create_2d_means(delta, k)
                        data = simulate_2d_data(means, true_partition, n)[1].tolist()
                    else: 
                        means = create_1d_means(delta, k)
                        data = simulate_data(means, true_partition, n)[1].tolist()
                    expression = f'{dim};{k};{delta};{n};{data};{means};{true_labels}\n'
                    file.write(expression)
    print('end')
    return


dimensions = [1, 2]
n_points = [25, 50, 75, 100, 250, 500, 750, 1000, 2000]
deltas = [0.25, 0.5, 0.75, 1, 3, 6]
n_clusters = [3, 5, 7, 11, 15]

# ## 1D data

simulate_datasets(deltas, n_clusters, n_points, 1)

# ## 2D data

simulate_datasets(deltas, n_clusters, n_points, 2)

# ### READ 1D

df_1 = pd.read_csv("Dataset_1d.txt", delimiter=';')
df_1.shape

df1

# +
#print(df1.isna().any().any()) 
# -

# ### READ 2D

df_2 = pd.read_csv("Dataset_2d.txt", delimiter=';')
df_2.shape

df_2.iloc[1:4]

# +
#print(df.isna().any().any()) 
# -






