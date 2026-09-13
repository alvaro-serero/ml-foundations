"""Calculate R-squared for Regression Analysis"""

import numpy as np

def r_squared(y_true, y_pred):
	mean = np.mean(y_true)

	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - mean) ** 2)

	r_squared = 1 - (ss_res / ss_tot)

	return round(r_squared, 3)


y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1, 2, 3, 4, 5])
print(r_squared(y_true, y_pred))

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
print(r_squared(y_true, y_pred))
