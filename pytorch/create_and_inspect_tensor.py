"""Create and Inspect a Tensor"""

import torch


def make_tensor():
    """Return a 2x3 float32 tensor [[1, 2, 3], [4, 5, 6]]."""
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)


t = make_tensor()
print(t)
print(t.shape)
print(t.dtype)
