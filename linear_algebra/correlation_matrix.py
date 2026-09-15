"""Calculate Correlation Matrix"""

import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X

    nb_samples, nb_features_x = X.shape
    nb_features_y = Y.shape[1]
    corr_matrix = np.zeros((nb_features_x, nb_features_y))

    # Means and standard deviations
    means_x, stds_x = X.mean(axis=0), X.std(axis=0)
    means_y, stds_y = Y.mean(axis=0), Y.std(axis=0)

    for i in range(nb_features_x):
        for j in range(nb_features_y):
            # Deviations
            diff_i = X[:, i] - means_x[i]
            diff_j = Y[:, j] - means_y[j]

            # Covariance
            covariance = np.sum(diff_i * diff_j) / nb_samples

            # Add to correlation matrix
            corr_matrix[i, j] = covariance / (stds_x[i] * stds_y[j])

    return corr_matrix


print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))
print(calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])))
print(calculate_correlation_matrix(np.array([[1, 0], [0, 1]]), np.array([[1, 2], [3, 4]])))


# Vectorized version of the function (without loops)
def calculate_correlation_matrix_vectorized(X, Y=None):
    if Y is None:
        Y = X

    n_samples = X.shape[0]

    # Center matrices, using the mean along each feature
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Pairwise covariances: (p, n) @ (n, q) -> (p, q)
    covariance = (X_centered.T @ Y_centered) / n_samples

    # Population std of each column
    std_X = X.std(axis=0)
    std_Y = Y.std(axis=0)

    # Outer product gives the (p, q) matrix of std_X[i] * std_Y[j]
    correlation = covariance / np.outer(std_X, std_Y)

    return correlation


print(calculate_correlation_matrix_vectorized(np.array([[1, 2], [3, 4], [5, 6]])))
print(calculate_correlation_matrix_vectorized(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])))
print(calculate_correlation_matrix_vectorized(np.array([[1, 0], [0, 1]]), np.array([[1, 2], [3, 4]])))
