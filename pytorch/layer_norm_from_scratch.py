"""Implement LayerNorm from Scratch"""

import torch
import torch.nn as nn


def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, correction=0)

    y = gamma * ((x - mean) / torch.sqrt(variance + eps)) + beta

    return y


# Test 1
torch.manual_seed(0)
x = torch.randn(2, 3, 8)
gamma = torch.ones(8); beta = torch.zeros(8)
out = layer_norm(x, gamma, beta)
print(round(out.mean(dim=-1).abs().max().item(), 4))

# Test 2
x = torch.randn(2, 3, 8)
gamma = torch.ones(8); beta = torch.zeros(8)
out = layer_norm(x, gamma, beta)
print(round(out.var(dim=-1, unbiased=False).mean().item(), 2))

# Test 3
x = torch.randn(4, 8)
ln_ref = nn.LayerNorm(8)
out_ref = ln_ref(x)
out_mine = layer_norm(x, ln_ref.weight.detach(), ln_ref.bias.detach())
print(torch.allclose(out_mine, out_ref, atol=1e-5))
