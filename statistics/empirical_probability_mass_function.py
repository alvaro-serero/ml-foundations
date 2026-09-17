"""Empirical Probability Mass Function (PMF)"""
from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    n = len(samples)
    if n == 0:
        return []
    
    counts = Counter(samples)
    return [(v, counts[v] / n) for v in sorted(counts)]


print(empirical_pmf([1, 2, 2, 3, 3, 3]))
print(empirical_pmf([5, 5, 5, 5]))
print(empirical_pmf([]))
print(empirical_pmf([0, 0, 1, 1, 1, 2]))
