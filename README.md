# bayesiancl
The package contains two methods of bayesian clustering based on projection theory.
- The *ProdForm* assumes the priors on the partitions are in product form and,
- the *CardBased* assumes the prior on the partitions varies based on the cardinality-array of the respective partition.

## Description
Bayesian Clustering is a technique using Bayesian Inference to cluster data based on its distribution.
It is assumed that the clusters are normally distributed with "true" and unknown means.
- The *target* of this method is to best partition the data into clusters.
- The number of cluster  is *NOT* assumed to be fixed but an upper bound k is given as input to the algorithm.

### This project includes:
1. Theory derived from Bayesian Inference.
2. The *ProdForm* and *CardBased* algorithms.
3. Performance evaluation distances
4. Simulated Datasets and results.
5. Application on the Iris Dataset.

The theory is addressed in the [Thesis Document](docs/Bayesian_Clustering_MSc.pdf).

## Visuals
Implementation of ProdForm on the iris dataset.

- Resulted partition when initialized with upper bound 3, against KMeans with predetermined number of clusters equal to 3.
![Clustering of ProdForm](notebooks/graphs/iris_K3.png)
- Resulted partition when initialized with upper bound 7 against KMeans with predetermined number of clusters equal to 7.
![Clustering of ProdForm](notebooks/graphs/iris_K7.png)
- Recovering the Normal distribution of the clusters
![Clustering of ProdForm](notebooks/graphs/iris_contours_posterior.png)
The priors are assumed to be the KMeans centers
## Installation of bayesiancl
Make sure Git is installed on your machine.

### 1. Clone the repository to your local machine:
```bash
git clone https://github.com/Australeza/bayesiancl.git
cd bayesiancl
```

### 2. Install the package
```bash
pip install .
```
Or in development mode (if you plan to modify the code):
```bash
pip install -e .
```
### 3. (Optional) Install dependencies
```bash
pip install -r requirements_dev.txt
```


## Usage
To see the project in action, check out the .py version of the [example notebook](notebooks/example-iris.py)

**It covers:**
- Application of ProdForm to 2-d data.
- Comparison between ProdForm method with KMeans and DBScan.
- Method behaviour based on different initializations.

## Authors and acknowledgment

**Elisavet Karanikola**

Master's student, [Applied Mathematics](https://vu.nl/en/about-vu/faculties/faculty-of-science/departments/mathematics)

elizkaranikola@gmail.com

## License
[MIT License](https://opensource.org/license/mit)

## Credits
This repository was developed during my internship at [**KPMG**](https://kpmg.com/nl/nl/home.html), as part of my Master's Thesis titled "Bayesian Clustering" submitted to [**Vrije Universiteit Amsterdam**](https://vu.nl/en) in [May 2025].

## Project status
#### Ongoing..
