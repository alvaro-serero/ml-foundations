"""Descriptive Statistics Calculator"""
import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    a = np.asarray(data, dtype=float)

    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    vals, counts = np.unique(a, return_counts=True)
    mode = vals[np.argmax(counts)]
    var = np.var(a)

    return {
        'mean': round(float(np.mean(a)), 4),
        'median': round(float(np.median(a)), 4),
        'mode': round(float(mode), 4),
        'variance': round(float(var), 4),
        'standard_deviation': round(float(np.sqrt(var)), 4),
        '25th_percentile': round(float(p25), 4),
        '50th_percentile': round(float(p50), 4),
        '75th_percentile': round(float(p75), 4),
        'interquartile_range': round(float(p75 - p25), 4),
    }


result = descriptive_statistics([1, 2, 2, 3, 4, 4, 4, 5])
print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})

result = descriptive_statistics([10, 20, 20, 30, 40])
print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})

result = descriptive_statistics([100])
print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})

result = descriptive_statistics([1, 1, 2, 2, 3, 3])
print(result['mode'])  # When multiple modes exist, return the smallest

result = descriptive_statistics([-5, 0, 5, 10, 15])
print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})
