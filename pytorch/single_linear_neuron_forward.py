"""Single Linear Neuron Forward"""

import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # TODO: build nn.Linear(3, 1), set fixed weight/bias under no_grad, return float output
    neuron = nn.Linear(3, 1)

    # Manually set parameters
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        neuron.bias.copy_(torch.tensor([0.1]))

    # Forward pass
    output = neuron(x)

    # Return output of forward pass as a float
    return output.item()


print(single_neuron_forward(torch.tensor([[1.0, 2.0, 3.0]])))
print(single_neuron_forward(torch.tensor([[0.0, 0.0, 0.0]])))
print(single_neuron_forward(torch.tensor([[1.0, 1.0, 1.0]])))
