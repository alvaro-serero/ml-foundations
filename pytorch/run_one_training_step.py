"""Run One Training Step: Forward, Loss, Backward, Optimizer"""

import torch
import torch.nn as nn

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.

    # Initialize SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x)

    # Compute MSE loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Return loss as a float
    return loss.item()

model = nn.Linear(1, 1)

with torch.no_grad():
    model.weight.fill_(1.0)
    model.bias.fill_(0.0)

x = torch.tensor([[2.0]])
y = torch.tensor([[6.0]])
lr = 0.1

loss = train_one_step(model, x, y, lr)

print(loss)
print(model.weight)
print(model.bias)


