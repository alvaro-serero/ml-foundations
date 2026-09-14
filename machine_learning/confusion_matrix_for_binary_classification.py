"""Generate a Confusion Matrix for Binary Classification"""

from collections import Counter

def confusion_matrix(data):
    counts = Counter(tuple(pair) for pair in data)

    tp = counts[(1, 1)]
    fn = counts[(1, 0)]
    fp = counts[(0, 1)]
    tn = counts[(0, 0)]

    return [[tp, fn], [fp, tn]]


data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
print(confusion_matrix(data))