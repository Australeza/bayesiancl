import numpy as np
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def perfect_match_distance(cluster_a:list[int], cluster_b:list[int])->int:
    """Perfect matching distance between two clusterings.

    Parameters
    ----------
    cluster_a: list
        clustering as 1d array of labels
    cluster_b: list
        clustering as 1d array of labels

    Returns
    -------
    int
        implements the hungarian algorithm to minimize the cost.
        Intuitively, it is minimum number of elements that need to be moved in
        clustering b to rememble clustering a.
        References:
        https://www.r-bloggers.com/2012/11/matching-clustering-solutions-using-the-hungarian-method/
    """
    #number of points in clusterings
    nA = len(cluster_a)
    nB = len(cluster_b)
    if nA != nB:
        raise ValueError("number of cluster or number of instances do not match")

    #
    #distinct clusters in clusterings
    n_clusters_a = list(set(cluster_a))
    n_clusters_b = list(set(cluster_b))

    #max cluster
    highest_cluster = max(max(n_clusters_a), max(n_clusters_b))
    keep_idxs = list(range(nA))

    # initialize cost matrix
    nC = highest_cluster+1
    cost_matrix = -np.ones(shape = (nC, nC), dtype=int)

    for i in range(nC):
        a_block = set([keep_idxs[j] for j in range(len(cluster_a)) if cluster_a[j] == i ])
        for j in range(nC):
            b_block = set([keep_idxs[k] for k in range(len(cluster_b)) if
                        cluster_b[k] == j])
            cost_matrix[i, j] = len(a_block.symmetric_difference(b_block))

    # The Hungarian algorithm (minimizes the total cost)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Total cost for matched blocks
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost



def variation_of_information(c:list[int], c_est:list[int])->[float,float]:
    r"""Compute variation of information.

    Parameters
    ----------
    c: list[int]
        clustering 1 - true
    c_est: list[int]
        clustering 2 - estimated
    k : int
        number of clusters
    Returns
    -------
    float, float
        Variation of Information score between X and Y.
        According to M. Meila, Journal of Multivariate Analysis 98 (2007) 873-895
        VI = Hc+Hc'-2*Icc', where Icc' mutual info score from sklearn
        Examples
             c = [1,0,0,0] and c_est = [1,0,0,0] then var_info = 0
             c = [1,0,0,0] and c_est = [0,1,1,1] then var_info = 0
        Remarks
        1.
        Adjusted variation of information is scaled between 0 and 1. (var_info/2*log(K*))
        K*, maximum number of clusters (the K* that minimizes the VI)
        Adjusted values close to 1 mean dissimilarity between clusters.
        2.
        It is advised to scale by 2*log(K*) when n varies and log(N) otherwise.
    """""
    #check lengths
    if len(c) != len(c_est):
        raise ValueError("vectors must have same length")

    #number of points
    n = len(c)

    #clusters
    clusters = sorted(list(set(c)))
    clusters_est = sorted(list(set(c_est)))

    #probability of abstract point belonging to cluster k, respectively
    P_k = np.array([c.count(k)/n for k in clusters])
    P_k_est = np.array([c_est.count(k)/n for k in clusters_est])

    #entropies
    H_c = - np.inner(P_k, np.log(P_k))
    H_c_est = - np.inner(P_k_est, np.log(P_k_est))

    #mutual information given by sklearn
    mutual_info = mutual_info_score(c, c_est)


    #variation of information
    var_info = H_c + H_c_est + - 2*mutual_info

    #adjusted variation of information with the maximum number of clusters
    max_clusters = max(len(c), len(c_est))
    print(max_clusters)
    adj_var_info = var_info/(2*np.log(max_clusters))
    return 1-adj_var_info



def evaluate_distances(true_x:np.array, estimated_x:np.array)->[float,float,int]:
    """Compute distance between true and estimated clusters.

    Parameters
    ----------
    true_x: np.array
        true labels
    estimated_x: np.array
        estimated labels

    Returns
    -------
    float, float, int
         Three different estimates between true and estimated partitions.
         Adjusted Rand Index, variation of information distance and perfect - matching.
         Adjusted Rand Index
          values close to 1, similar clusterings
                 close to 0, dissimilarity between clusterings
                 -0.5<values<0 clusterings similarity is worse than random
        Variation of Information-adjusted
            values close to 1, dissimilarity
            values close to 0, similarity
        Perfect matching
            Integer, number of elements to be moved from one to resemble the other.
                The lower the better, ideally, 0.

    """
    rand_distance = adjusted_rand_score(true_x, estimated_x)
    vi_distance = variation_of_information(true_x, estimated_x)
    perfect_matching = perfect_match_distance(true_x, estimated_x)
    return rand_distance, vi_distance, perfect_matching
