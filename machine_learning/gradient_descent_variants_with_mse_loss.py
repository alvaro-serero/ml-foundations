"""Implement Gradient Descent Variants with MSE Loss"""

import numpy as np


def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights
    """
    m = len(y)

    # Start training loop
    for epoch in range(n_epochs):
        if method == "batch":
            y_hat = X @ weights
            errors = y_hat - y
            gradient_loss = (2 / m) * X.T @ errors
            weights = weights - learning_rate * gradient_loss

        elif method == "stochastic":
            for i in range(m):
                x_i = X[i]
                y_i = y[i]
                y_hat = x_i @ weights
                error = y_hat - y_i
                gradient_loss = 2 * x_i * error
                weights = weights - learning_rate * gradient_loss
        elif method == "mini_batch":
            for start in range(0, m, batch_size):
                # create X_batch and y_batch
                X_batch = X[start:start + batch_size]
                y_batch = y[start:start + batch_size]
                # prediction
                y_hat = X_batch @ weights
                # error
                error = y_hat - y_batch
                # calculate current batch size
                current_batch_size = len(y_batch)
                # gradient
                gradient_loss = (2 / current_batch_size) * (X_batch.T @ error)
                # update
                weights = weights - learning_rate * gradient_loss

    return weights


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])
weights = np.zeros(X.shape[1])
learning_rate = 0.01
n_epochs = 100

# Test Batch Gradient Descent
output = gradient_descent(X, y, weights, learning_rate, n_epochs, method='batch')
print(output)
