# bayesiancl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-completed-orange.svg)](#project-status)
[![Interactive demo](https://img.shields.io/badge/demo-interactive%20explorer-2ea44f.svg)](https://australeza.github.io/bayesiancl/explorer.html)

**Bayesian clustering without fixing the number of clusters.** `bayesiancl` partitions data with Bayesian inference: you give an upper bound $k$ on the number of clusters, and the method finds the most probable partition of the data, which may use fewer clusters.

It implements two methods, built on projection theory, that differ in the prior they place on partitions:

| Method | Class | Prior on partitions |
| :--- | :--- | :--- |
| **ProdForm** | `ProductForm` | Product form: a product of point-level priors. |
| **CardBased** | `CardBased` | Depends on the partition's cardinality array (the sizes of its clusters). |

---

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [API overview](#api-overview)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Author and credits](#author-and-credits)
- [License](#license)

---

## How it works

- Each cluster is assumed to be normally distributed around an unknown "true" mean.
- You supply **prior means**, one per possible cluster. Their number sets the upper bound $k$. For example, they can be K-Means centres.
- The model combines these priors with the data to compute posterior probabilities over partitions, then returns the **maximum a posteriori (MAP) partition**.
- 1-D and 2-D data are supported.

The full theory (derivations, assumptions and proofs) is in the [thesis](docs/Bayesian_Clustering_MSc.pdf).

---

## Installation

Requires Python 3.10+ and [Git](https://git-scm.com/).

```bash
git clone https://github.com/Australeza/bayesiancl.git
cd bayesiancl
pip install -r requirements.txt   # dependencies
pip install -e .                  # the package itself
```

Optional, for development (linting and pre-commit hooks):

```bash
pip install -r requirements_dev.txt
```

> [!NOTE]
> The package is imported as `bayesian_clustering`, even though the repository is called `bayesiancl`.

---

## Quickstart

```python
import numpy as np
from bayesian_clustering import ProductForm

# 1-D toy data: three groups around 1, 4.5 and 10
X = np.array([1.2, 1.4, 4.5, 4.6, 10.0, 11.0])

# One prior mean per possible cluster -> upper bound k = 3
prior_means = np.array([1.0, 5.0, 10.0])

model = ProductForm(sampling_size=300)
model.fit(X, prior_means=prior_means)
labels, partition = model.predict(X)

print(labels)     # [0, 0, 1, 1, 2, 2]  cluster label per point
print(partition)  # [[1, 2], [3, 4], [5, 6]]  points per cluster (1-indexed)
```

`CardBased` follows the same `fit` / `predict` pattern.

For a full worked example on the Iris dataset, with K-Means centres as priors, see [`notebooks/example-iris.py`](notebooks/example-iris.py). It is a Jupytext script that can be opened as a notebook.

---

## API overview

Both classes share the same interface:

| Method | Description |
| :--- | :--- |
| `fit(data, prior_means)` | Computes the marginals and posterior quantities. `prior_means` has shape `(k,)` for 1-D data or `(k, 2)` for 2-D data. |
| `predict(data)` | Returns the MAP partition. |
| `fit_predict(data, prior_means)` | Runs `fit` and then `predict`. |
| `assumptions_partition_prior_check()` | Checks that the partition-prior assumptions hold. |

Main constructor options:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `use_partition_space` | `False` | If `True`, enumerates **every** partition to find the exact MAP. This is only feasible for small datasets, because the number of partitions grows very fast. |
| `sampling_size` | `None` | If set, the method samples this many partitions from the point-wise posterior and returns the most frequent one. If `None`, it takes the most probable cluster for each point. |

`ProductForm.fit` also accepts `partition_prior_distribution`: either a `scipy.stats` distribution or a NumPy array of point-level probabilities. The default is a uniform product prior.

---

## Results

### Iris dataset

> 🔎 **[Try the interactive explorer](https://australeza.github.io/bayesiancl/explorer.html)**: change the upper bound $k$ with a slider and hover over points to inspect them.

**ProdForm** on Iris (petal area vs. sepal area), using K-Means centres as prior means:

<p align="center">
  <img src="notebooks/graphs/iris_K3.png" width="900" alt="ProdForm, K-Means and true labels on Iris with k=3">
</p>

<p align="center">
  <img src="notebooks/graphs/iris_K7.png" width="900" alt="ProdForm, K-Means and true labels on Iris with k=7">
</p>

Colours are matched to the true species, so the same group has the same colour in every panel.

Everything in this section is reproducible from the repository (run from the repo root):

```bash
python notebooks/make_readme_figures.py   # the two figures above
python notebooks/make_explorer_data.py    # data for the interactive explorer (docs/explorer_data.json)
```

### Recovered normal distributions

Recovered cluster distributions, with contours of their normal PDFs around the posterior centroids. Assumption: the centroids are d-close to each other, where the separation distance satisfies d:k(d).

<p align="center">
  <img src="notebooks/graphs/iris_contours_posterior.png" width="500" alt="Posterior contours of the recovered clusters">
</p>

### Simulated data

Benchmarks on simulated 1-D and 2-D datasets are in [`notebooks/simulations/`](notebooks/simulations/), and the resulting plots are in [`notebooks/graphs/`](notebooks/graphs/). Methods are compared using the distance metrics in `bayesian_clustering/distances.py`.

---

## Repository structure

```
bayesian_clustering/      # the package
├── algorithms/           # ProductForm and CardBased classes
├── priors.py             # partition and point-level priors
├── product_form.py       # ProdForm computations
├── card_based.py         # CardBased computations
├── prediction_post.py    # posterior prediction and sampling
├── distances.py          # evaluation metrics
└── data_simulation.py    # simulated data generation
notebooks/                # Iris example, theory notebooks, simulations, plots
docs/                     # thesis PDF and datasets
```

---

## Author and credits

**Elisavet Karanikola**: MSc Applied Mathematics, [Vrije Universiteit Amsterdam](https://vu.nl/en)
[elizkaranikola@gmail.com](mailto:elizkaranikola@gmail.com)

Developed during an internship at [**KPMG**](https://kpmg.com/nl/nl/home.html), as part of the Master's thesis *"Bayesian Clustering"*, submitted to Vrije Universiteit Amsterdam in May 2025.

### Acknowledgements

The theoretical framework behind this package builds on the unpublished notes *Posterior analysis for k-groups partitions* by **prof. dr. E. N. Belitser** (Vrije Universiteit Amsterdam) [[1]](#references), who supervised the thesis. I am grateful to him for sharing that work and for his guidance throughout the project. I also thank dr. P. J. de Andrade Serra for acting as second reader, and R. van der Wal and the Advanced Analytics & Big Data team at KPMG for hosting the internship.

### References

[1] E. N. Belitser. *Posterior analysis for k-groups partitions*. Unpublished notes, Vrije Universiteit Amsterdam.

References cited in the thesis:

[2] F. Abramovich, Y. Benjamini, D. L. Donoho and I. M. Johnstone. Adapting to unknown sparsity by controlling the false discovery rate. *The Annals of Statistics*, 34(2):584–653, 2006.

[3] D. L. Banks and S. E. Fienberg. Statistics, Multivariate. In R. A. Meyers (ed.), *Encyclopedia of Physical Science and Technology* (3rd ed.), pp. 851–889. Academic Press, 2003. [doi:10.1016/B0-12-227410-5/00731-6](https://doi.org/10.1016/B0-12-227410-5/00731-6)

[4] E. Belitser and S. Ghosal. Empirical Bayes oracle uncertainty quantification for regression. *The Annals of Statistics*, 48(6):3113–3137, 2020.

[5] E. Belitser and N. Nurushev. Local posterior concentration rate for multilevel sparse sequences. In *Bayesian Statistics in Action: BAYSM 2016, Florence, Italy*, pp. 51–66. Springer, 2017.

[6] E. Belitser and N. Nurushev. Needles and straw in a haystack: Robust confidence for possibly sparse sequences. 2020.

[7] E. Belitser, N. Nurushev and P. Serra. Robust estimation of sparse signal with unknown sparsity cluster value. In *Conference of the International Society for Non-Parametric Statistics*, pp. 77–87. Springer, 2018.

[8] I. Castillo and A. W. van der Vaart. Supplement to "Needles and straw in a haystack: Posterior concentration for possibly sparse sequences". *The Annals of Statistics*, 40(4), 2012.

[9] A. K. Jain. Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8):651–666, 2010. [doi:10.1016/j.patrec.2009.09.011](https://doi.org/10.1016/j.patrec.2009.09.011)

[10] M. Meilă. Comparing clusterings: an information based distance. *Journal of Multivariate Analysis*, 98(5):873–895, 2007.

[11] P. Nerurkar, A. Shirke, M. Chandane and S. Bhirud. Empirical analysis of data clustering algorithms. *Procedia Computer Science*, 125:770–779, 2018. [doi:10.1016/j.procs.2017.12.099](https://doi.org/10.1016/j.procs.2017.12.099)

[12] B. Szabó and A. van der Vaart. *Bayesian Statistics*. Lecture notes, version 20-7-2023, 2023.

[13] T. Velmurugan and T. Santhanam. A survey of partition based clustering algorithms in data mining: An experimental approach. *Information Technology Journal*, 10(3):478–484, 2011.

[14] S. Wade and Z. Ghahramani. Bayesian cluster analysis: Point estimation and credible balls (with discussion). *Bayesian Analysis*, 13(2):559–626, 2018.

[15] KPMG. KPMG official page. 2025. https://kpmg.com/nl/nl/home/over-ons.html (visited 20 April 2025).

BibTeX for [1]:

```bibtex
@unpublished{belitser_kgroups,
  author = {Belitser, Eduard N.},
  title  = {Posterior analysis for k-groups partitions},
  note   = {Unpublished notes, Vrije Universiteit Amsterdam}
}
```

---

## License

Released under the [MIT License](https://opensource.org/license/mit).

---

## Project status

Completed.
