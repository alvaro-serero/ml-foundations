"""Reshape and Transpose a Tensor"""

import torch

def reshape_transpose(t):
    """Reshape a 1D tensor of 6 elements to 2x3 (row-major) and return its transpose.

    Args:
        t (torch.Tensor): 1D tensor with exactly 6 elements.

    Returns:
        torch.Tensor: Transpose of the 2x3 reshape, with shape (3, 2).
    """
    # TODO: reshape to (2, 3) then transpose to (3, 2)
    return t.reshape(2, 3).T


print(reshape_transpose(torch.tensor([1, 2, 3, 4, 5, 6])))
print(reshape_transpose(torch.arange(6)))
