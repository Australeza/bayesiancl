import numpy as np
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from scipy.optimize import linear_sum_assignment

def count_mismatches(true_labels:list[int], aligned_labels:list[int]) ->int:
    """Count the number of mismatches between true labels and aligned labels.

    Parameters
    ----------
    true_labels:list
        true partition in labels
    aligned_labels:list
        estimated partition in labels

    Returns
    -------
    int
        The number of mismatches

    """
    mismatches = np.sum(np.array(true_labels) != aligned_labels)
    return mismatches

def best_map(true_labels:list, pred_labels:list)->float:
    """
    Permutes pred_labels to match true_labels as closely as possible.

    Parameters
    ----------
    true_labels :list
        Ground truth labels.
    pred_labels : list
        Predicted labels (from clustering).

    Returns
    -------
    new_pred_labels : np.ndarray
        Relabeled version of pred_labels that best aligns with true_labels.
        Refer to "https://github.com/indigits/sparse-plex", in MATLAB.
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)

    n_true = unique_true.size
    n_pred = unique_pred.size
    n_classes = max(n_true, n_pred)

    # Build the cost matrix
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_val in enumerate(unique_true):
        for j, pred_val in enumerate(unique_pred):
            matches = np.sum((true_labels == true_val) & (pred_labels == pred_val))
            cost_matrix[i, j] = -matches  # Negative for maximization

    # Solve the assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping from pred to true
    label_mapping = {}
    for i, j in zip(row_ind, col_ind):
        if i < n_true and j < n_pred:
            label_mapping[unique_pred[j]] = unique_true[i]

    # Remap labels
    new_pred_labels = np.vectorize(lambda x: label_mapping.get(x, x))(pred_labels)
    n_miss = count_mismatches(true_labels, new_pred_labels)
    return 1 - n_miss/len(true_labels)


def variation_of_information(c: list[int], c_est: list[int]) -> float:
    """
    Computes adjusted Variation of Information (VI) between two clusterings.
    Returns a value between 0 (similar) and 1 (dissimilar).
    """
    if len(c) != len(c_est):
        raise ValueError("Both clusterings must be of the same length")

    n = len(c)

    # Build distributions
    labels_c = np.array(c)
    labels_c_est = np.array(c_est)

    clusters = np.unique(labels_c)
    clusters_est = np.unique(labels_c_est)

    P_c = np.array([np.sum(labels_c == label) / n for label in clusters])
    P_c_est = np.array([np.sum(labels_c_est == label) / n for label in clusters_est])

    H_c = -np.sum(P_c * np.log(P_c))
    H_c_est = -np.sum(P_c_est * np.log(P_c_est))

    mutual_info = mutual_info_score(labels_c, labels_c_est)

    vi = H_c + H_c_est - 2 * mutual_info

    # Normalize with respect to max possible entropy
    k_star = max(len(clusters), len(clusters_est))
    norm_vi = vi / (2 * np.log(k_star)) if k_star > 1 else 0

    # Return similarity measure: 1 means identical, 0 means completely dissimilar
    return 1 - norm_vi



def evaluate_distances(true_x:np.array, estimated_x:np.array)->[float,float,float]:
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
        Three different performance scores between true and estimated partitions.
        Adjusted Rand Index, Variation of Information-adjusted, Perfect matching
        In all distances:
            values close to 1, similarity
            values close to 0, dissimilarity

    """
    adj_rand_distance = adjusted_rand_score(true_x, estimated_x)
    adj_vi_distance = variation_of_information(true_x, estimated_x)
    adj_pmd = best_map(true_x, estimated_x)
    return adj_rand_distance, adj_vi_distance, adj_pmd
