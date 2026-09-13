"""Linear Regression Using Normal Equation"""

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X)
	y = np.array(y)
	theta = np.linalg.inv(X.T @ X) @ X.T @ y
	return np.round(theta, 4).tolist()


print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))
print(linear_regression_normal_equation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1]))
print(linear_regression_normal_equation([[1, 0], [1, 1], [1, 2]], [2, 2, 2]))
print(linear_regression_normal_equation([[1, 2], [1, 4], [1, 6]], [3, 5, 7]))
