"""Implement BatchNorm2d from Scratch (training mode)"""

import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    c = x.shape[1]
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    variance = x.var(dim=(0, 2, 3), keepdim=True, correction=0)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)

    gamma_reshaped = gamma.reshape((1, c, 1, 1))
    beta_reshaped = beta.reshape((1, c, 1, 1))

    y = gamma_reshaped * x_normalized + beta_reshaped

    return y


torch.manual_seed(0)
x = torch.randn(8, 4, 6, 6)
gamma = torch.ones(4)
beta = torch.zeros(4)
out = batchnorm2d(x, gamma, beta)
per_ch_mean = out.mean(dim=(0, 2, 3))
print([round(v, 4) for v in per_ch_mean.tolist()])
