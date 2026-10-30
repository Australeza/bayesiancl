"""Regenerate the Iris comparison figures used in the README.

Run from the repository root:
    python notebooks/make_readme_figures.py

Outputs:
    notebooks/graphs/iris_K3.png
    notebooks/graphs/iris_K7.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from bayesian_clustering import ProductForm

OUT = Path(__file__).parent / "graphs"
PALETTE = plt.get_cmap("tab10").colors

# --- Data: [petal area, sepal area] -------------------------------------
iris = load_iris()
data, true_labels = iris.data, iris.target
X = np.c_[data[:, 2] * data[:, 3], data[:, 0] * data[:, 1]]


def align_labels(labels, reference):
    """Relabel clusters so they best match the reference labels.

    Cluster ids are arbitrary, so without this the same group gets a
    different colour in each panel. Extra clusters keep distinct ids.
    """
    labels = np.asarray(labels)
    ids = np.unique(labels)
    ref_ids = np.unique(reference)
    overlap = np.array([[np.sum((labels == i) & (reference == j)) for j in ref_ids] for i in ids])
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {ids[r]: ref_ids[c] for r, c in zip(rows, cols)}
    next_id = len(ref_ids)
    for i in ids:
        if i not in mapping:
            mapping[i] = next_id
            next_id += 1
    return np.array([mapping[i] for i in labels])


def cluster(k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    np.random.seed(42)  # ProductForm sampling uses numpy's global RNG
    model = ProductForm(sampling_size=500).fit(X, prior_means=kmeans.cluster_centers_)
    prodform_labels, _ = model.predict(X)
    return prodform_labels, kmeans.labels_


def plot(k, filename):
    prodform, kmeans = cluster(k)
    panels = [
        (f"ProdForm ({len(np.unique(prodform))} clusters found)", align_labels(prodform, true_labels)),
        (f"K-Means ({k} clusters)", align_labels(kmeans, true_labels)),
        ("True species", true_labels),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True, facecolor="white")
    for ax, (title, labels) in zip(axes, panels):
        ax.scatter(X[:, 0], X[:, 1], c=[PALETTE[i % 10] for i in labels],
                   s=22, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sepal area (cm$^2$)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Petal area (cm$^2$)", fontsize=11)
    fig.suptitle(f"Iris clustering with upper bound $k = {k}$", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT / filename}")


if __name__ == "__main__":
    plot(3, "iris_K3.png")
    plot(7, "iris_K7.png")
