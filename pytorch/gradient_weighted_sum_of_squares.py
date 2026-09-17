"""Gradient of a Weighted Sum of Squares"""

import torch


def grad_wss(w_list, x_list):
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    w = torch.tensor(w_list, dtype=torch.float32, requires_grad=True)
    x = torch.tensor(x_list, dtype=torch.float32)

    # L = 1/2 * sum((w_i * x_i)^2)
    loss = 0.5 * torch.sum((w * x) ** 2)

    # Computation of dL/dw
    loss.backward()

    return [round(val, 4) for val in w.grad.tolist()]


print(grad_wss([1.0, 2.0], [3.0, 4.0]))
print(grad_wss([0.5, -1.0, 2.0], [2.0, 3.0, 1.0]))
print(grad_wss([-2.0, 3.0, 0.5, 1.0], [1.0, 2.0, 4.0, 0.5]))
