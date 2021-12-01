# import modules 
import numpy as np


def extrapolate_dataset_distribution(data): 
    # create index 
    data = data.set_index(['dataset'])
    data = data.sort_index()

    # get multivariate mean and covariance for each multi-index pair 
    indices = list(set(data.index))
    indices.sort()
    vector_means, vector_covariances = [], []
    for dataset in indices: 
        vectors = list(data.loc[dataset, 
            'vectorized_text'])
        vectors = np.array(vectors)
        mean = np.mean(vectors, axis=0)
        mean = np.reshape(mean, (mean.shape[0], 1))
        vector_means.append(mean)
        vector_covariances.append(np.cov(vectors, rowvar=False))
    vector_means = np.stack(vector_means, axis=0)
    vector_covariances = np.stack(vector_covariances, axis=0)

    # return mean and covariance information
    return indices, vector_means, vector_covariances

def extrapolate_sentiment_distribution(data): 
    # create multi-index 
    data = data.set_index(['dataset', 'sentiment'])
    data = data.sort_index()

    # get multivariate mean and covariance for each multi-index pair 
    multi_indices = list(set(data.index))
    multi_indices.sort()
    vector_means, vector_covariances = [], []
    for dataset, sentiment in multi_indices: 
        vectors = list(data.loc[(dataset, sentiment), 
            'vectorized_text'])
        vectors = np.array(vectors)
        mean = np.mean(vectors, axis=0)
        mean = np.reshape(mean, (mean.shape[0], 1))
        vector_means.append(mean)
        vector_covariances.append(np.cov(vectors, rowvar=False))
    vector_means = np.stack(vector_means, axis=0)
    vector_covariances = np.stack(vector_covariances, axis=0)

    # return mean and covariance information
    return multi_indices, vector_means, vector_covariances

def _compute_bhattacharyya_distance(mean1, mean2, covariance1, covariance2):
    # compute first term of Bhattacharyya distance 
    covariance = (covariance1 + covariance2) / 2
    det_covariance = np.linalg.det(covariance)
    det_covariance1 = np.linalg.det(covariance1)
    det_covariance2 = np.linalg.det(covariance2)
    bhattacharyya_distance_term1 = 1 / 2 * np.log(det_covariance 
        / np.sqrt(det_covariance1 * det_covariance2))

    # compute second term of Bhattacharyya distance 
    mean_diff = mean2 - mean1
    bhattacharyya_distance_term2 = 1 / 8 * np.matmul(
        np.matmul(
            np.transpose(mean_diff), 
            np.linalg.pinv(covariance)
        ), 
        mean_diff 
    )

    # computer Bhattacharyya distance from terms 
    bhattacharyya_distance = (bhattacharyya_distance_term1 
        + bhattacharyya_distance_term2)
    
    # return Bhattacharyya distance 
    return bhattacharyya_distance

def compute_adjusted_bhattacharyya_distances(indices, means, covariances, 
    adjustment_factor=1):
    # compute bhattacharyya distances for each pair of datasets 
    bhattacharyya_distances = np.zeros((len(indices), len(indices)))
    for i in range(len(indices)):
        for j in range(len(indices)):
            mean1, mean2 = means[i], means[j]
            covariance1 = covariances[i] * adjustment_factor
            covariance2 = covariances[j] * adjustment_factor
            bhattacharyya_distances[i, j] = _compute_bhattacharyya_distance(
                mean1, mean2, covariance1, covariance2)
    
    # return computed Bhattacharyya distances 
    return bhattacharyya_distances