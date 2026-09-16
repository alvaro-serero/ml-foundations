"""One Training Step"""

import torch
import torch.nn as nn


def train_step(model, x, y, optimizer, loss_fn):
    """Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """
    # TODO: zero_grad -> forward -> loss -> backward -> step; return float loss

    # Zero out optimizer's gradients
    optimizer.zero_grad()

    # Forward pass
    pred = model(x)

    # Loss
    loss = loss_fn(pred, y)

    # Backward pass
    loss.backward()

    # Step
    optimizer.step()

    # Return float loss
    return float(loss.item())


# Test 1
model = nn.Linear(2, 1)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[0.5, -0.3]]))
    model.bias.copy_(torch.tensor([0.1]))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])
print(round(train_step(model, x, y, opt, loss_fn), 4))

# Test 2
model = nn.Linear(2, 1)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[0.5, -0.3]]))
    model.bias.copy_(torch.tensor([0.1]))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])
train_step(model, x, y, opt, loss_fn)
print(round(train_step(model, x, y, opt, loss_fn), 4))