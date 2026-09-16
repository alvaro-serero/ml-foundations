"""Gradient of a Square with Autograd"""

import torch

def grad_of_square(x_val):
    """Return dy/dx for y = x**2 at x = x_val using autograd.

    Args:
        x_val (float): scalar input value.

    Returns:
        float: gradient of x**2 w.r.t. x at x_val.
    """
    # TODO: create tensor, compute y = x**2, backward, return grad
    # Create a pytorch tensor from x_val with gradient activated
    x = torch.tensor(x_val, requires_grad=True)

    # Forward pass
    y = x ** 2

    # Backward pass
    y.backward()

    # Get the gradient as a float
    return x.grad.item()


print(grad_of_square(3.0))
print(grad_of_square(2.5))
print(grad_of_square(-1.0))
