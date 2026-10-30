"""Generate the data shown on the interactive explorer (docs/explorer.html).

Runs the bayesian_clustering package on the Iris dataset for every upper
bound k in K_RANGE and writes the results to docs/explorer_data.json, which
the explorer page loads. Everything displayed on the page comes from here.

Run from the repository root:
    python notebooks/make_explorer_data.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

from bayesian_clustering import ProductForm
from make_readme_figures import align_labels

K_RANGE = range(2, 8)    # upper bounds shown on the slider
SAMPLING_SIZE = 500      # ProductForm posterior samples
SEED = 42                # ProductForm sampling uses numpy's global RNG

OUT = Path(__file__).resolve().parent.parent / "docs" / "explorer_data.json"

# --- Data: [petal area, sepal area], as in example-iris.py -----------------
iris = load_iris()
data, true_labels = iris.data, iris.target
X = np.c_[data[:, 2] * data[:, 3], data[:, 0] * data[:, 1]]


def run(k):
    """Cluster with K-Means and with ProductForm (K-Means centres as priors)."""
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    np.random.seed(SEED)
    model = ProductForm(sampling_size=SAMPLING_SIZE).fit(X, prior_means=kmeans.cluster_centers_)
    prodform, _ = model.predict(X)
    prodform = np.asarray(prodform)
    return {
        "prodform": align_labels(prodform, true_labels).tolist(),
        "kmeans": align_labels(kmeans.labels_, true_labels).tolist(),
        "prodform_k": int(len(np.unique(prodform))),
        "ari_pf": round(float(adjusted_rand_score(true_labels, prodform)), 3),
        "ari_km": round(float(adjusted_rand_score(true_labels, kmeans.labels_)), 3),
        "priors": kmeans.cluster_centers_.round(3).tolist(),
    }


if __name__ == "__main__":
    results = {
        "settings": {"sampling_size": SAMPLING_SIZE, "seed": SEED},
        "x": X[:, 0].round(3).tolist(),
        "y": X[:, 1].round(3).tolist(),
        "species": [str(iris.target_names[i]) for i in true_labels],
        "true": true_labels.tolist(),
        "runs": {},
    }
    for k in K_RANGE:
        results["runs"][k] = run(k)
        r = results["runs"][k]
        print(f"k={k}: ProdForm found {r['prodform_k']} clusters, "
              f"ARI ProdForm {r['ari_pf']:.2f}, ARI K-Means {r['ari_km']:.2f}")
    OUT.write_text(json.dumps(results))
    print(f"saved {OUT}")
