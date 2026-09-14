"""Calculate Dice Score for Classification"""

import numpy as np

def dice_score(y_true, y_pred):
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))
	denominator = 2 * tp + fp + fn

	if denominator == 0:
		return 0.0

	dice_score = (2 * tp) / denominator

	return round(dice_score, 3)


y_true = np.array([1, 1, 0, 0])
y_pred = np.array([1, 1, 0, 0])
print(dice_score(y_true, y_pred))

y_true = np.array([1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0])
print(dice_score(y_true, y_pred))

y_true = np.array([0, 0, 0, 0])
y_pred = np.array([0, 0, 0, 0])
print(dice_score(y_true, y_pred))
