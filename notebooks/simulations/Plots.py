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

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("./simulations/Results_correct_2", sep=";")#df = pd.read_csv("Dataset_2", sep=";")
df.columns

df[df["clusters"]==5][:5]

unique_values_dict = {col: df[col].unique() for col in df.columns}
#unique_values_dict

# +
import matplotlib.pyplot as plt
import seaborn as sns

# Filter the DataFrame
df_only_20 = df[df['delta'] == 6]

# Plot setup
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Plot each method
sns.lineplot(data=df_only_20, x="sample_size", y="kmeans_metric_vi", label="KMeans", marker='o')
sns.lineplot(data=df_only_20, x="sample_size", y="db_metric_vi", label="DBSCAN", marker='s')
sns.lineplot(data=df_only_20, x="sample_size", y="prodform_metric_vi", label="Product Form", marker='^')

# Add titles and labels
plt.title("Rand Index vs Sample Size ($\\delta = 0.2$)", fontsize=14)
plt.xlabel("Sample Size", fontsize=12)
plt.ylabel("Rand Index", fontsize=12)
plt.legend(title="Method")
plt.tight_layout()
plt.show()

# -

delta_values = [0.2, 0.8, 1.1, 2.5]
df.sample_size.unique()

# +

#figure size
plt.figure(figsize=(12, 5)) 

df_long = df.melt(
    id_vars=['sample_size', 'delta'],
    value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'db_metric_vi'],
    var_name='Method',
    value_name='Performance'
)


df_long['Method'] = df_long['Method'].replace({
    'kmeans_metric_vi': 'K-Means',
    'prodform_metric_vi': 'Product Form',
    'db_metric_vi': 'DBScan'
})

# Plot with Seaborn
sns.lineplot(data=df_long, x='sample_size', y='Performance', hue='Method')

plt.title("Performance Comparison of K=3 clusters")
plt.xlabel("Delta")
plt.ylabel("Performance metric VI")
#plt.xticks(deltas)
plt.grid(True)
plt.tight_layout()
plt.show()

# +
plt.figure(figsize=(12, 5))
df_2000 = df[df["clusters"]==3]

# Melt the DataFrame
df_long = df_2000.reset_index().melt(
    id_vars='delta',
    value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'db_metric_vi'],
    var_name='Method',
    value_name='Performance'
)

sns.lineplot(data=df_long, x='delta', y='Performance', hue='Method', marker='o', ci=None)

plt.title("Performance vs Delta at Different Sample Sizes K=3")
plt.xlabel("Delta")
plt.ylabel("Performance")
plt.grid(False)
plt.tight_layout()
plt.show()

# -

df.columns

# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define cluster values to plot
cluster_values = [3, 7, 11]

# Filter the DataFrame to only include delta > 1
df_filtered = df[df["delta"] >=2.5]

# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, k in enumerate(cluster_values):
    ax = axes[i]
    df_k = df_filtered[df_filtered["clusters"] == k]

    # Reshape to long format
    df_long = df_k.reset_index().melt(
        id_vars='sample_size',
        value_vars=['kmeans_metric_perf', 'prodform_metric_perf', 'dbscan_metric_perf'],
        var_name='Method',
        value_name='Performance'
    )

    # Plot on the current axis
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='o',
        ax=ax,
        palette="viridis"
    )

    ax.set_title(f"Performance vs Sample Size (K={k}, Δ > 1)")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Performance (VI)")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(False)

# Add legend to the right
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower right', title="Method")

plt.tight_layout(rect=[0, 0, 0.93, 1])  # leave space for legend
plt.show()


# +
# Define cluster values to plot
cluster_values = [3, 5,7]

# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, k in enumerate(cluster_values):
    ax = axes[i]
    df_k = df[df["clusters"] == k]

    # Reshape to long format
    df_long = df_k.reset_index().melt(
        id_vars='sample_size',
        value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'dbscan_metric_vi'],
        var_name='Method',
        value_name='Performance'
    )

    df_long['Method'] = df_long['Method'].replace({
        'kmeans_metric_vi': 'KMeans',
        'prodform_metric_vi': 'ProdForm',
        'dbscan_metric_vi': 'DBScan'
    })

    # Plot on the current axis
    sns.lineplot(
        data=df_long,
        x='sample_size',  # Corrected x-axis
        y='Performance',
        hue='Method',
        marker='o',
        ci=None,
        ax=ax,
        palette="viridis"
    )

    ax.set_title(f"VI vs Sample Size (K={k})")
    ax.set_xlabel("Sample Size")

    if i == 0:
        ax.set_ylabel("Performance (VI)")
        handles, labels = ax.get_legend_handles_labels()  # Only capture legend once
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add global legend
fig.legend(handles, labels, loc='center right', title="Method")

# Layout adjustment
plt.tight_layout(rect=[0, 0, 0.93, 1])  # Reserve space for the legend
plt.savefig('plot_VI_N_1d.png', dpi=300, bbox_inches='tight')
plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.8, 1.1, 2.5,]  # Example deltas, you can customize this

# Create the figure and axes
fig, axes = plt.subplots(1, len(delta_values), figsize=(18, 5), sharey=True)

# Loop through each delta value to create subplots
for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Filter for the current delta
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format
    df_long = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'db_metric_vi'],
        var_name='Method',
        value_name='Performance'
    )

    # Plot on the current axis
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='o',
        ax=ax,
        palette="viridis"
    )

    # Set title, labels and adjust axes
    ax.set_title(f"Performance vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Performance (VI)")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add shared legend to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Method")

plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the legend
plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.2, 1.1, 2.5, 6]
df = df[df["clusters"]==7]
# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format
    df_long = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'dbscan_metric_vi'],
        var_name='Method',
        value_name='Performance'
    )

    # Clean up method names for the legend
    df_long['Method'] = df_long['Method'].replace({
        'kmeans_metric_rand': 'KMeans',
        'prodform_metric_rand': 'ProdForm',
        'dbscan_metric_rand': 'DBScan'
    })

    # Plot on the current axis
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='o',
        ci=None,
        ax=ax,
        palette="viridis"
    )

    ax.set_title(f"Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Performance (Rand Index)")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add shared legend to the right
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', title="Method")

plt.tight_layout(rect=[0, 0, 0.93, 1])  # leave space on the right for legend
plt.savefig('plot_sample_size_methods.png', dpi=300, bbox_inches='tight')
plt.show()


# +

import matplotlib.pyplot as plt
import seaborn as sns

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 5), sharey=True)

# ---- Plot 1: VI metric vs sample_size ----
df_vi_sample = df.reset_index().melt(
    id_vars='sample_size',
    value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'db_metric_vi'],
    var_name='Method',
    value_name='Performance'
)
sns.lineplot(data=df_vi_sample, x='sample_size', y='Performance', hue='Method', ax=axes[0])
axes[0].set_title("Performance (VI) vs Sample Size")
axes[0].set_xlabel("Sample Size")
axes[0].set_ylabel("Performance (VI)")
axes[0].grid(True)

# ---- Plot 2: Rand Index vs sample_size ----
df_rand_sample = df.reset_index().melt(
    id_vars='sample_size',
    value_vars=['kmeans_metric_rand', 'prodform_metric_rand', 'db_metric_rand'],
    var_name='Method',
    value_name='Performance'
)
sns.lineplot(data=df_rand_sample, x='sample_size', y='Performance', hue='Method', ax=axes[1])
axes[1].set_title("Rand Index vs Sample Size")
axes[1].set_xlabel("Sample Size")
axes[1].set_ylabel("Rand Index")
axes[1].grid(True)

# ---- Plot 3: Perf vs sample_size ----
df_perf_sample = df.reset_index().melt(
    id_vars='sample_size',
    value_vars=['kmeans_metric_perf', 'prodform_metric_perf', 'dbscan_metric_perf'],
    var_name='Method',
    value_name='Performance'
)
sns.lineplot(data=df_perf_sample, x='sample_size', y='Performance', hue='Method', ax=axes[2])
axes[2].set_title("Performance (Perf) vs Sample Size")
axes[2].set_xlabel("Sample Size")
axes[2].set_ylabel("Performance (Perf)")
axes[2].grid(True)

# Final layout and display
plt.tight_layout()
plt.show()


# +
import matplotlib.pyplot as plt
import pandas as pd

cluster_values = [3, 5, 7]

# Metric columns to compare
metrics = {
    'kmeans_metric_rand': 'Rand Index',
    'kmeans_metric_vi': 'VI',
    'kmeans_metric_perf': 'Perf'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')

# ---- Row 1: Metric vs Sample Size ----
for col_idx, (metric_col, metric_name) in enumerate(metrics.items()):
    ax = axes[0, col_idx]
    for k in cluster_values:
        df_k = df[df["clusters"] == k]
        mean_values = df_k.groupby("sample_size")[metric_col].mean().reset_index()
        ax.plot(mean_values["sample_size"], mean_values[metric_col], marker='o', label=f"K={k}")

    ax.set_title(f"{metric_name} vs Sample Size")
    ax.set_xlabel("Sample Size")
    if col_idx == 0:
        ax.set_ylabel("Performance")
    ax.grid(True)
    ax.legend(title="Clusters (K)")

# ---- Row 2: Metric vs Delta ----
for col_idx, (metric_col, metric_name) in enumerate(metrics.items()):
    ax = axes[1, col_idx]
    for k in cluster_values:
        df_k = df[df["clusters"] == k]
        mean_values = df_k.groupby("delta")[metric_col].mean().reset_index()
        ax.plot(mean_values["delta"], mean_values[metric_col], marker='o', label=f"K={k}")

    ax.set_title(f"{metric_name} vs Delta")
    ax.set_xlabel("Delta")
    if col_idx == 0:
        ax.set_ylabel("Performance")
    ax.grid(True)
    ax.legend(title="Clusters (K)")

plt.tight_layout()
plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.8, 1.1, 2.5, 6]

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Filter for current delta
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format with all cluster values
    df_long = df_d.melt(
        id_vars=['sample_size', 'clusters'],
        value_vars=['kmeans_metric_rand', 'prodform_metric_rand', 'dbscan_metric_rand'],
        var_name='Method',
        value_name='Performance'
    )

    # Clean up method names for clarity
    df_long['Method'] = df_long['Method'].replace({
        'kmeans_metric_rand': 'KMeans',
        'prodform_metric_rand': 'ProdForm',
        'dbscan_metric_rand': 'DBScan'
    })

    # Plot with style encoding for different cluster numbers
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='Method',
        style='clusters',  # differentiate by number of clusters
        markers=True,
        dashes=False,
        ax=ax,
        palette="viridis"
    )

    ax.set_title(f"Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add global legend outside
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Method / Clusters (K)")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
plt.savefig('plot_RandIndex_by_Delta_with_K.png', dpi=300, bbox_inches='tight')
plt.show()

# -



# +
cluster_values = [3, 5, 7]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ---- Plot 1: kmeans_metric_rand vs sample_size ----
for k in cluster_values:
    df_k = df[df["clusters"] == k]
    mean_values = df_k.groupby("delta")["kmeans_metric_vi"].mean().reset_index()
    axes[0].plot(mean_values["delta"], mean_values["kmeans_metric_vi"], marker='o', label=f"K={k}")

axes[0].set_title("Rand Index vs Sample Size")
axes[0].set_xlabel("Sample Size")
axes[0].set_ylabel("Rand Index")
axes[0].grid(True)
axes[0].legend(title="Clusters")

# ---- Plot 2: kmeans_metric_rand vs delta ----
for k in cluster_values:
    df_k = df[df["clusters"] == k]
    mean_values = df_k.groupby("sample_size")["kmeans_metric_rand"].mean().reset_index()
    axes[1].plot(mean_values["sample_size"], mean_values["kmeans_metric_rand"], marker='o', label=f"K={k}")

axes[1].set_title("Rand Index vs Delta")
axes[1].set_xlabel("Delta")
axes[1].grid(True)
axes[1].legend(title="Clusters")

plt.tight_layout()
plt.show()

# -



dimensions = [1, 2]
n_points = [20, 50, 80, 100, 150, 200, 300, 500, 800, 1000, 2000]
deltas = [0.2, 0.5, 0.8, 1.1, 1.5, 2.5, 6]
n_clusters = [3, 5, 7, 11, 15, 21]

# ### This one

# +
sample_sizes = [20, 100, 500]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, n in enumerate(sample_sizes):
    ax = axes[i]
    df_filtered = df[
        (df["sample_size"] == n) &
        (df["clusters"].isin([3, 5, 7])) &
        (df["delta"] < 7.5)
    ]
    df_filtered = df_filtered.sort_values(["clusters", "delta"])

    sns.lineplot(
        data=df_filtered,
        x="delta",
        y="prodform_metric_vi",
        hue="clusters",
        marker="o",
        ax=ax,
        palette="Dark2"
    )

    ax.set_title(f"Rand Index vs Delta (N={n})")
    ax.set_xlabel("Delta")
    if i == 0:
        ax.set_ylabel("Adj VI Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Shared legend on the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Clusters (K)")
plt.savefig('../thesis_adjvi_delta_prodform.png', dpi=300, bbox_inches='tight')
plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.show()

# -



df = pd.read_csv("Results_Dataset_2", sep=";")


# +
import seaborn as sns
import matplotlib.pyplot as plt

sample_sizes = [20, 100, 500]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, n in enumerate(sample_sizes):
    ax = axes[i]
    df_filtered = df[
        (df["sample_size"] == n) &
        (df["clusters"].isin([3, 5, 7, 11])) &
        (df["delta"] < 7.5)
    ]
    df_filtered = df_filtered.sort_values(["clusters", "delta"])

    sns.lineplot(
        data=df_filtered,
        x="delta",
        y="kmeans_metric_rand",
        hue="clusters",
        marker="o",
        ax=ax,
        palette="Dark2"
    )

    ax.set_title(f"Rand Index vs Delta (N={n})")
    ax.set_xlabel("Delta")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Shared legend on the right
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', title="Clusters (K)")
plt.savefig('prodform-randindex.png', dpi=300, bbox_inches='tight')
plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.show()


# +
sample_sizes = [20, 100, 100]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, n in enumerate(sample_sizes):
    ax = axes[i]
    df_filtered = df[
        (df["sample_size"] == n) &
        (df["clusters"].isin([3, 5, 7, 11])) &
        (df["delta"] < 7.5)
    ]
    df_filtered = df_filtered.sort_values(["clusters", "delta"])

    sns.lineplot(
        data=df_filtered,
        x="delta",
        y="kmeans_metric_vi",
        hue="clusters",
        marker="o",
        ax=ax,
        palette="Dark2"
    )

    ax.set_title(f"Adj VI vs Delta (N={n})")
    ax.set_xlabel("Delta")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Shared legend on the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Clusters (K)")

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.show()


# +
import seaborn as sns
import matplotlib.pyplot as plt

sample_sizes = [20,100, 500]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, n in enumerate(sample_sizes):
    ax = axes[i]
    df_filtered = df[
        (df["sample_size"] == n) &
        (df["clusters"].isin([3, 5, 7, 11])) &
        (df["delta"] < 7.5)
    ]
    df_filtered = df_filtered.sort_values(["clusters", "delta"])

    sns.lineplot(
        data=df_filtered,
        x="delta",
        y="prodform_metric_perf",
        hue="clusters",
        marker="o",
        ax=ax,
        palette="Dark2"
    )

    ax.set_title(f"Adj. PMD vs Delta (N={n})")
    ax.set_xlabel("Delta")
    if i == 0:
        ax.set_ylabel("Adj. PMD")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Shared legend on the right
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', title="Clusters (K)")
plt.savefig('plot_pefect.png', dpi=300, bbox_inches='tight')

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.show()

# -



# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.2, 0.8, 2.5]

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Filter for current delta
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format with all cluster values
    df_long = df_d.melt(
        id_vars=['sample_size', 'clusters'],
        value_vars=[ 'prodform_metric_rand'],
        var_name='Method',
        value_name='Performance'
    )

    # Clean up method names for clarity
    df_long['Method'] = df_long['Method'].replace({
        'kmeans_metric_rand': 'KMeans',
        'prodform_metric_rand': 'ProdForm',
    })

    # Plot with style encoding for different cluster numbers
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='Method',
        style='clusters',  # differentiate by number of clusters
        markers=True,
        dashes=False,
        ax=ax,
        palette="viridis"
    )

    ax.set_title(f"Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add global legend outside
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Method / Clusters (K)")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
plt.savefig('plot_RandIndex_by_Delta_with_K.png', dpi=300, bbox_inches='tight')
plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.8, 1.1, 2.5, 6]

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Filter for current delta
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format with all cluster values
    df_long = df_d.melt(
        id_vars=['sample_size', 'clusters'],
        value_vars=['prodform_metric_rand'],
        var_name='Method',
        value_name='Performance'
    )

    df_long['Method'] = df_long['Method'].replace({
        'prodform_metric_rand': 'ProdForm'
    })

    # Plot with color encoding for different clusters
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='clusters',      # color by number of clusters
        marker='o',
        ax=ax,
        palette='viridis'
    )

    ax.set_title(f"Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add global legend for clusters
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Number of Clusters (K)")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
plt.savefig('plot_RandIndex_by_Delta_clusters_colored.png', dpi=300, bbox_inches='tight')
plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.8, 2.5, 6]

# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Filter for current delta
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format with all cluster values
    df_long = df_d.melt(
        id_vars=['sample_size', 'clusters'],
        value_vars=['prodform_metric_vi'],
        var_name='Method',
        value_name='Performance'
    )

    df_long['Method'] = df_long['Method'].replace({
        'prodform_metric_vi': 'ProdForm'
    })

    # Plot with color encoding for different clusters
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='clusters',      # color by number of clusters
        marker='o',
        ax=ax,
        palette='viridis'
    )

    ax.set_title(f"Adj. VI vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Adj. VI")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add global legend for clusters
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', title="Number of Clusters (K)")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
plt.savefig('plot_RandIndex_by_Delta_clusters_colored.png', dpi=300, bbox_inches='tight')
plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.8, 1.1, 2.5, 6]

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Filter for current delta
    df_d = df[(df["delta"] == delta)&(df["sample_size"]<500)]
    
    # Reshape to long format with all cluster values
    df_long2 = df_d.melt(
        id_vars=['sample_size', 'clusters'],
        value_vars=['prodform_metric_perf'],
        var_name='Method',
        value_name='Performance'
    )

    df_long2['Method'] = df_long2['Method'].replace({
        'prodform_metric_perf': 'ProdForm'
    })

    # Plot with color encoding for different clusters
    sns.lineplot(
        data=df_long2,
        x='sample_size',
        y='Performance',
        hue='clusters',      # color by number of clusters
        marker='o',
        ax=ax,
        palette='viridis'
    )

    ax.set_title(f"Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add global legend for clusters
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Number of Clusters (K)")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
plt.savefig('plot_RandIndex_by_Delta_clusters_colored.png', dpi=300, bbox_inches='tight')
plt.show()

# -

# ### This one

# +
df_1000 = df[df["sample_size"] == 1000]
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

methods = {
    'kmeans_metric_vi': 'KMeans',
    'prodform_metric_vi': 'ProdForm',
    'dbscan_metric_vi': 'DBSCAN'
}

for i, (col, name) in enumerate(methods.items()):
    ax = axes[i]
    df_long = df_1000[['delta', 'clusters', col]].rename(columns={col: 'Performance'})
    df_long['Method'] = name

    sns.lineplot(
        data=df_long,
        x='delta',
        y='Performance',
        hue='clusters',
        marker='o',
        palette='viridis',
        ax=ax
    )
    ax.set_title(f"{name} Performance vs Delta (Sample=1000)")
    ax.set_xlabel("Delta")
    if i == 0:
        ax.set_ylabel("Adj VI Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()
    ax.grid(True)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', title='Clusters (K)')
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('adj_vi_sample1000_methods.png', dpi=300, bbox_inches='tight')
plt.show()

# -



# +
df_25 = df[df["delta"] == 2.5]

import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

methods = {
    'kmeans_metric_perf': 'KMeans',
    'prodform_metric_perf': 'ProdForm',
    'dbscan_metric_perf': 'DBSCAN'
}

for i, (col, name) in enumerate(methods.items()):
    ax = axes[i]
    df_long = df_25[['sample_size', 'clusters', col]].rename(columns={col: 'Performance'})
    df_long['Method'] = name

    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='clusters',
        marker='o',
        palette='viridis',
        ax=ax
    )
    ax.set_title(f"{name} Performance vs Sample Size (Δ=2.5)")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()
    ax.grid(True)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title='Clusters (K)')
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('performance_vs_sample_size_delta_2.5.png', dpi=300, bbox_inches='tight')
plt.show()

# -

# ### This one

# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.2, 1.1, 2.5, 6]
df = df[df["clusters"] == 7]  # Filter for clusters = 7

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

for i, delta in enumerate(delta_values):
    ax = axes[i]
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format
    df_long = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=['kmeans_metric_vi', 'prodform_metric_vi', 'dbscan_metric_vi'],
        var_name='Method',
        value_name='Performance'
    )

    # Clean up method names for the legend
    df_long['Method'] = df_long['Method'].replace({
        'kmeans_metric_vi': 'KMeans',
        'prodform_metric_vi': 'ProdForm',
        'dbscan_metric_vi': 'DBScan'
    })

    # Plot on the current axis
    sns.lineplot(
        data=df_long,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='o',
        ci=None,
        ax=ax,
        palette="viridis"
    )

    ax.set_title(f"Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Performance (Rand Index)")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add shared legend to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Method")

plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for legend
plt.savefig('plot_sample_size_methods.png', dpi=300, bbox_inches='tight')
plt.show()

# -



# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.2, 1.1, 2.5, 6]
df = df[df["clusters"] == 7]  # Filter for clusters = 7

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

# Define the metrics to plot
metrics = ['kmeans_metric_vi', 'prodform_metric_vi', 'dbscan_metric_vi']
rand_metrics = ['kmeans_metric_rand', 'prodform_metric_rand', 'dbscan_metric_rand']

for i, delta in enumerate(delta_values):
    ax = axes[i]
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format for Performance (VI)
    df_long_perf = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=metrics,
        var_name='Method',
        value_name='Performance'
    )

    # Reshape to long format for Rand Index
    df_long_rand = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=rand_metrics,
        var_name='Method',
        value_name='Rand Index'
    )

    # Clean up method names for the legend
    df_long_perf['Method'] = df_long_perf['Method'].replace({
        'kmeans_metric_vi': 'KMeans',
        'prodform_metric_vi': 'ProdForm',
        'dbscan_metric_vi': 'DBScan'
    })
    
    df_long_rand['Method'] = df_long_rand['Method'].replace({
        'kmeans_metric_rand': 'KMeans',
        'prodform_metric_rand': 'ProdForm',
        'dbscan_metric_rand': 'DBScan'
    })

    # Plot Performance (VI) for each method
    sns.lineplot(
        data=df_long_perf,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='o',
        ci=None,
        ax=ax,
        palette="viridis"
    )

    # Plot Rand Index (as a second line) with different color scheme
    sns.lineplot(
        data=df_long_rand,
        x='sample_size',
        y='Rand Index',
        hue='Method',
        marker='x',
        ci=None,
        ax=ax,
        palette="coolwarm"
    )

    ax.set_title(f"Performance & Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Performance / Rand Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()

    ax.grid(True)

# Add shared legend to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Method")

plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for legend
plt.savefig('plot_sample_size_methods_with_rand_perf.png', dpi=300, bbox_inches='tight')
plt.show()

# -



# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define delta values to plot
delta_values = [0.2, 1.1, 2.5, 6]
df = df[df["clusters"] == 7]  # Filter for clusters = 7

# Create the figure and axes
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

# Define the metrics to plot
metrics = ['kmeans_metric_vi', 'prodform_metric_vi', 'dbscan_metric_vi']
rand_metrics = ['kmeans_metric_rand', 'prodform_metric_rand', 'dbscan_metric_rand']
perf_metrics = ['kmeans_metric_perf', 'prodform_metric_perf', 'dbscan_metric_perf']

# Define custom color palette
color_palette = {
    'KMeans': 'tab:blue',       # Blue for KMeans
    'ProdForm': 'tab:orange',   # Orange for ProdForm
    'DBScan': 'tab:green'       # Green for DBScan
}

for i, delta in enumerate(delta_values):
    ax = axes[i]
    df_d = df[df["delta"] == delta]
    
    # Reshape to long format for Performance (VI)
    df_long_perf = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=metrics,
        var_name='Method',
        value_name='Performance'
    )

    # Reshape to long format for Rand Index
    df_long_rand = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=rand_metrics,
        var_name='Method',
        value_name='Rand Index'
    )

    # Reshape to long format for Performance
    df_long_perf_metric = df_d.reset_index().melt(
        id_vars='sample_size',
        value_vars=perf_metrics,
        var_name='Method',
        value_name='Performance'
    )

    # Clean up method names for the legend
    df_long_perf['Method'] = df_long_perf['Method'].replace({
        'kmeans_metric_vi': 'KMeans',
        'prodform_metric_vi': 'ProdForm',
        'dbscan_metric_vi': 'DBScan'
    })
    
    df_long_rand['Method'] = df_long_rand['Method'].replace({
        'kmeans_metric_rand': 'KMeans',
        'prodform_metric_rand': 'ProdForm',
        'dbscan_metric_rand': 'DBScan'
    })
    
    df_long_perf_metric['Method'] = df_long_perf_metric['Method'].replace({
        'kmeans_metric_perf': 'KMeans',
        'prodform_metric_perf': 'ProdForm',
        'dbscan_metric_perf': 'DBScan'
    })

    # Plot Performance (VI) for each method (same color)
    sns.lineplot(
        data=df_long_perf,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='o',
        ci=None,
        ax=ax,
        palette=color_palette,
        legend=False  # We will add the legend manually later
    )

    # Plot Rand Index (as a second line) with same color
    sns.lineplot(
        data=df_long_rand,
        x='sample_size',
        y='Rand Index',
        hue='Method',
        marker='x',
        ci=None,
        ax=ax,
        palette=color_palette,
        legend=False  # We will add the legend manually later
    )

    # Plot Performance (from new metric) with same color
    sns.lineplot(
        data=df_long_perf_metric,
        x='sample_size',
        y='Performance',
        hue='Method',
        marker='s',
        ci=None,
        ax=ax,
        palette=color_palette,
        legend=False  # We will add the legend manually later
    )

    ax.set_title(f"Performance & Rand Index vs Sample Size (Δ={delta})")
    ax.set_xlabel("Sample Size")
    if i == 0:
        ax.set_ylabel("Performance / Rand Index")
    else:
        ax.set_ylabel("")
    
    ax.grid(True)

# Add shared legend to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title="Method")

plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for legend
plt.savefig('plot_sample_size_methods_with_rand_perf_same_colors.png', dpi=300, bbox_inches='tight')
plt.show()

# -





df=pd.read_csv("Results_CardBased_25_3_1d.txt", sep=";")#df = pd.read_csv("Dataset_2", sep=";")
df.columns

# +
plt.figure(figsize=(12, 5))
df_2000 = df[df["clusters"]==3]

# Melt the DataFrame
df_long = df_2000.reset_index().melt(
    id_vars='delta',
    value_vars=['kmeans_avi', 'prodform_avi', 'dbscan_avi'],
    var_name='Method',
    value_name='Performance'
)

sns.lineplot(data=df_long, x='delta', y='Performance', hue='Method', marker='o', ci=None)

plt.title("Performance vs Delta at Different Sample Sizes K=3")
plt.xlabel("Delta")
plt.ylabel("Performance")
plt.grid(False)
plt.tight_layout()
plt.show()


# +
df_1000 = df
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

methods = {
    'kmeans_avi': 'KMeans',
    'prodform_avi': 'ProdForm',
    'dbscan_avi': 'DBSCAN'
}

for i, (col, name) in enumerate(methods.items()):
    ax = axes[i]
    df_long = df_1000[['delta', 'clusters', col]].rename(columns={col: 'Performance'})
    df_long['Method'] = name

    sns.lineplot(
        data=df_long,
        x='delta',
        y='Performance',
        hue='clusters',
        marker='o',
        palette='viridis',
        ax=ax
    )
    ax.set_title(f"{name} Performance vs Delta (Sample=1000)")
    ax.set_xlabel("Delta")
    if i == 0:
        ax.set_ylabel("Adj VI Index")
    else:
        ax.set_ylabel("")
        ax.legend().remove()
    ax.grid(True)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', title='Clusters (K)')
plt.tight_layout(rect=[0, 0, 0.9, 1])
#plt.savefig('adj_vi_sample1000_methods.png', dpi=300, bbox_inches='tight')
plt.show()

# -






