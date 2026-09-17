"""Expected Value and Variance of an n-Sided Die"""

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    s = sum(i for i in range(1, n + 1))
    # E[X] = sum(​k * (1/n)) ​= (1/n) * (n(n+1)/2) ​= (n+1) / 2​
    expected_value = (n + 1) / 2
    # Sum of squares formula
    variance = (n ** 2 - 1) / 12
    return (expected_value, variance)
