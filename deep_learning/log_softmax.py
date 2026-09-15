"""Implementation of Log Softmax Function"""

import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores_shifted = scores - np.max(scores)
    return scores_shifted - np.log(np.sum(np.exp(scores_shifted)))


print(np.round(log_softmax([1, 2, 3]), 4))
print(np.round(log_softmax([1, 1, 1]), 4))
print(np.round(log_softmax([1, 1, .0000001]), 4))
